#!/usr/bin/python

import pandas as pd
import numpy as np
import pickle
import sys
import os

def predict_proba(year, mileage, State, Make, Model):

   clf = pickle.load(open(os.path.dirname(__filename__) + '/rf_model_max2.pkl','rb')) 

   Ft1 = pd.DataFrame([[year, mileage]], columns=['year','mileage'])

 # Create features
   State1= 'State_ '+State
   FtS= pd.DataFrame(np.zeros((1, 51)), columns=['State_ AK', 'State_ AL', 'State_ AR', 'State_ AZ',
    'State_ CA', 'State_ CO', 'State_ CT', 'State_ DC', 'State_ DE',
    'State_ FL', 'State_ GA', 'State_ HI', 'State_ IA', 'State_ ID',
    'State_ IL', 'State_ IN', 'State_ KS', 'State_ KY', 'State_ LA',
    'State_ MA', 'State_ MD', 'State_ ME', 'State_ MI', 'State_ MN',
    'State_ MO', 'State_ MS', 'State_ MT', 'State_ NC', 'State_ ND',
    'State_ NE', 'State_ NH', 'State_ NJ', 'State_ NM', 'State_ NV',
    'State_ NY', 'State_ OH', 'State_ OK', 'State_ OR', 'State_ PA',
    'State_ RI', 'State_ SC', 'State_ SD', 'State_ TN', 'State_ TX',
    'State_ UT', 'State_ VA', 'State_ VT', 'State_ WA', 'State_ WI',
    'State_ WV', 'State_ WY'])
 
   FtS.loc[0,State1]=1
 
 
   Make1= 'make1_'+Make
   FtM = pd.DataFrame(np.zeros((1, 6)), columns=['make1_Chevrolet', 'make1_Ford',
    'make1_Honda', 'make1_Jeep', 'make1_Others', 'make1_Toyota'])
   FtM.loc[0,Make1]=1
  
   Model1= 'model1_'+Model   
   Ftmod = pd.DataFrame(np.zeros((1, 8)), columns=['model1_Accord', 'model1_Civic', 'model1_F-1504WD', 'model1_Grand', 'model1_Others', 'model1_Sierra', 'model1_Silverado', 'model1_Wrangler'])
   
   Ftmod.loc[0,Model1]=1
 
   FtT=pd.concat([Ft1, FtS, FtM,Ftmod], axis=1,)

                                             
                          
 # Make prediction
   p1 = clf.predict(FtT)[0]

   return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please enter car features')
        
    else:
       
        print('Precio estimado: ')
        