import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PowerTransformer, RobustScaler

#read data
test = pd.read_csv("test_data.csv")
train = pd.read_csv("train_data.csv")
val = pd.read_csv("val_data.csv")

#define features
hemoglobin = train['hemoglobin'].to_numpy()
dental = train['dental caries'].to_numpy()
fasting = train['fasting blood sugar'].to_numpy()
LDL = train['LDL'].to_numpy()
height = train['height(cm)'].to_numpy()
weight = train['weight(kg)'].to_numpy()
serum = train['serum creatinine'].to_numpy()
gtp = train['Gtp'].to_numpy()

hemoglobin_test = test['hemoglobin'].to_numpy()
dental_test = test['dental caries'].to_numpy()
fasting_test = test['fasting blood sugar'].to_numpy()
LDL_test = test['LDL'].to_numpy()
height_test = test['height(cm)'].to_numpy()
weight_test = test['weight(kg)'].to_numpy()
serum_test = test['serum creatinine'].to_numpy()
gtp_test = test['Gtp'].to_numpy()

hemoglobin_val = val['hemoglobin'].to_numpy()
dental_val = val['dental caries'].to_numpy()
fasting_val = val['fasting blood sugar'].to_numpy()
LDL_val = val['LDL'].to_numpy()
height_val = val['height(cm)'].to_numpy()
weight_val = val['weight(kg)'].to_numpy()
serum_val = val['serum creatinine'].to_numpy()
gtp_val = val['Gtp'].to_numpy()


features_train = [hemoglobin, fasting, LDL, height, weight, serum, gtp]
features_test = [hemoglobin_test, fasting_test, LDL_test, height_test, weight_test, serum_test, gtp_test]
features_val = [hemoglobin_val, fasting_val, LDL_val, height_val, weight_val, serum_val, gtp_val]

def get_tranformations(feature):
    # square root transformation of the feature
    sqaure_root_transformed_data = np.sqrt( feature)

    # log transformation of the feature  
    trf = FunctionTransformer(np.log1p)
    log_transformed_data = trf.fit_transform(feature)

    # Box-Cox transformation of the feature
    pt = PowerTransformer(method='box-cox')
    box_cox_transformed_data = pt.fit_transform(feature.reshape(-1, 1))

    #Yeo-Johnson transformation of the feature
    yt = PowerTransformer(method='yeo-johnson') #variation of box-cox but can handle negative values
    yeo_johnson_transformed_data = yt.fit_transform(feature.reshape(-1, 1))


    return sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data


def get_normalized(features):
    sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[0])
    #scaling the data using standard scaler
    scaler = StandardScaler()
    scaler.fit(box_cox_transformed_data)
    scaled_hemoglobin_train = scaler.transform(box_cox_transformed_data)


    sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[1])
    #scaling the data using standard scaler
    scaler = StandardScaler()
    scaler.fit(box_cox_transformed_data)
    scaled_fasting_blood_sugar_train = scaler.transform(box_cox_transformed_data)

    #scaling the data using standard scaler usnig original data
    scaler = StandardScaler()
    LDL_feature = features[2].reshape(-1, 1)
    scaler.fit(LDL_feature)
    scaled_LDL_train = scaler.transform(LDL_feature)

    #scaling the data using min-max scaler using original data
    scaler = MinMaxScaler()
    height_feature = features[3].reshape(-1, 1)
    scaler.fit(height_feature)
    scaled_height_train = scaler.transform(height_feature)

    #scaling the data using robust scaler beacause it is robust to outliers and can handle them
    scaler = RobustScaler()
    weight_feature = features[4].reshape(-1, 1)
    scaler.fit(weight_feature)
    scaled_weight = scaler.transform(weight_feature)

    sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[5])
    log_transformed_data = log_transformed_data.reshape(-1, 1)
    #scaling the data using min-max scaler
    scaler = MinMaxScaler()
    scaler.fit(log_transformed_data)
    scaled_serum_creatinine = scaler.transform(log_transformed_data)

    sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[6])
    #scaling the data using standard scaler
    scaler = StandardScaler()
    scaler.fit(box_cox_transformed_data)
    scaled_Gtp = scaler.transform(box_cox_transformed_data)

    return scaled_hemoglobin_train, scaled_fasting_blood_sugar_train, scaled_LDL_train, scaled_height_train, scaled_weight, scaled_serum_creatinine, scaled_Gtp

def get_normalized_train():
    return get_normalized(features_train)


def get_normalized_test():
    return get_normalized(features_test)

def get_normalized_val():
    return get_normalized(features_val)




def print_tranformations():
   pass

