"""
Created on Fri Mar 16 21:06:35 2018

@author: Ivana Hybenoa
"""

import pickle
from flask import Flask, request, make_response, send_file
from flasgger import Swagger
import pandas as pd
import zipfile
import time
from io import BytesIO


with open('/model/final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict_file', methods=["POST"])
def predict_file():
    """Example file endpoint returning a prediction
    ---
    parameters:
      - name: file_input_test
        in: formData
        type: file
        required: true
    responses:
        200:
            description: OK
    """
    df = pd.read_csv(request.files.get("file_input_test"))
    
    # Data preprocessing
    
    # Missing basement values should be 0
    df['basement'] = df.basement.fillna(0)
    
    # 'shake-shingle' and 'asphalt,shake-shingle' should be 'Shake Shingle'
    df.roof.replace(['shake-shingle', 'asphalt,shake-shingle'], 'Shake Shingle', inplace=True)

    # 'Rock, Stone' should be 'Masonry'
    df.exterior_walls.replace('Rock, Stone', 'Masonry', inplace=True)

    # 'Concrete' and 'Block' should be 'Concrete Block'
    df.exterior_walls.replace(['Concrete', 'Block'], 'Concrete Block', inplace=True)
    
    # Remove lot_size outliers
    df = df[df.lot_size <= 500000]

    # Fill missing categorical values
    for column in df.select_dtypes(include=['object']):
        df[column] = df[column].fillna('Missing')
        
    # Feature engineering
    # Create indicator variable for properties with 2 beds and 2 baths
    df['two_and_two'] = ((df.beds == 2) & (df.baths == 2)).astype(int) 
    
    # Create indicator feature for transactions between 2010 and 2013, inclusive
    df['during_recession'] = ((df.tx_year >= 2010) & (df.tx_year <= 2013)).astype(int)
    
    # Create a property age feature
    df['property_age'] = df.tx_year - df.year_built
    
    # However, for this problem, we are only interested in houses that already exist because the REIT only buys existing ones!
    df = df[df.property_age >= 0]
    
    # Create a school score feature that num_schools * median_school
    df['school_score'] = df.num_schools * df.median_school
    
    # Group 'Wood Siding' and 'Wood Shingle' with 'Wood'
    df.exterior_walls.replace(['Wood Siding', 'Wood Shingle'], 'Wood', inplace = True )
    
    # List of classes to group
    other_exterior_walls = ['Concrete Block', 'Stucco', 'Masonry', 'Other', 'Asbestos shingle']
    # Group other classes into 'Other'
    df.exterior_walls.replace(other_exterior_walls, 'Other', inplace = True)
    
    df.roof.replace(['Composition', 'Wood Shake/ Shingles'], 'Composition Shingle', inplace = True)
    # List of classes to group
    other_roof = ['Other', 'Gravel/Rock', 'Roll Composition', 'Slate', 'Built-up', 'Asbestos', 'Metal']
    # Group other classes into 'Other'
    df.roof.replace(other_roof, 'Other', inplace = True)
    
    # Create new dataframe with dummy features
    df = pd.get_dummies(df, columns = ['exterior_walls', 'roof', 'property_type'], drop_first = True)
    
    # Drop 'tx_year' and 'year_built' from the dataset
    df = df.drop(['tx_year', 'year_built'], axis = 1) # axis=1, because we are dropping columns
    
    prediction = model.predict(df)
    df['probability'] = pd.DataFrame(prediction_probability)
    data = df
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    input_data.to_excel(writer, sheet_name='predictions', 
                        encoding='urf-8', index=False)
    writer.save()
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        names = ['predictions.xlsx'] # names = ['file1.xlsx', 'file2.xlsx']
        files = [output]  # files = [output, output2]
        for i in range(len(files)):
            input_data = zipfile.ZipInfo(names[i])
            input_data.date_time = time.localtime(time.time())[:6]
            input_data_compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(input_data, files[i].getvalue())
    memory_file.seek(0)
    
    response = make_response(send_file(memory_file, attachment_filename = 'predictions.zip',
                                       as_attachment=True))
    response.headers['Content-Disposition'] = 'attachment;filename=predictions.zip'
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    
