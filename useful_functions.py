import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PowerTransformer, RobustScaler


def sqrt_transform(feature):
    return np.sqrt(feature)

def log_transform(feature):
    trf = FunctionTransformer(np.log1p)
    return trf.fit_transform(feature)

def box_cox_transform(feature):
    pt = PowerTransformer(method='box-cox')
    return pt.fit_transform(feature.reshape(-1, 1)).reshape(-1)

def yee_jhonson_transform(feature):
    yt = PowerTransformer(method='yeo-johnson') #variation of box-cox but can handle negative values
    return yt.fit_transform(feature.reshape(-1, 1)).reshape(-1)


def get_normalized_data(data):
    train_data = pd.read_csv("train_data.csv")

    # Hemoglobin
    scaler_hemoglobin = StandardScaler()
    hemoglobin_feature = box_cox_transform(data['hemoglobin'].to_numpy()).reshape(-1,1)                        
    hemoglobin_train = box_cox_transform(train_data['hemoglobin'].to_numpy()).reshape(-1,1)
    scaler_hemoglobin.fit(hemoglobin_train)
    scaled_hemoglobin = scaler_hemoglobin.transform(hemoglobin_feature)

    # Fasting Blood Sugar
    scaler_fasting = StandardScaler()
    fasting_feature = box_cox_transform(data['fasting blood sugar'].to_numpy()).reshape(-1,1)
    fasting_train = box_cox_transform(train_data['fasting blood sugar'].to_numpy()).reshape(-1,1)
    scaler_fasting.fit(fasting_train)
    scaled_fasting = scaler_fasting.transform(fasting_feature)

    # LDL
    scaler_ldl = StandardScaler()
    LDL_feature = data['LDL'].to_numpy().reshape(-1,1)
    LDL_train = train_data['LDL'].to_numpy().reshape(-1,1)
    scaler_ldl.fit(LDL_train)
    scaled_LDL = scaler_ldl.transform(LDL_feature)

    # Height
    scaler_height = MinMaxScaler()
    height_feature = data['height(cm)'].to_numpy().reshape(-1,1)
    height_train = train_data['height(cm)'].to_numpy().reshape(-1,1)
    scaler_height.fit(height_train)
    scaled_height = scaler_height.transform(height_feature)

    # Weight
    scaler_weight = RobustScaler()
    weight_feature = data["weight(kg)"].to_numpy().reshape(-1,1)
    weight_train = train_data["weight(kg)"].to_numpy().reshape(-1,1)
    scaler_weight.fit(weight_train)
    scaled_weight = scaler_weight.transform(weight_feature)

    # Serum
    scaler_serum = MinMaxScaler()
    serum_feature = log_transform(data["serum creatinine"].to_numpy()).reshape(-1,1)
    serum_train = log_transform(train_data["serum creatinine"].to_numpy()).reshape(-1,1)
    scaler_serum.fit(serum_train)
    scaled_serum = scaler_serum.transform(serum_feature)

    # GTP
    scaler_gtp = StandardScaler()
    gtp_feature = box_cox_transform(data['Gtp'].to_numpy()).reshape(-1,1)
    gtp_train = box_cox_transform(train_data['Gtp'].to_numpy()).reshape(-1,1)
    scaler_gtp.fit(gtp_train)
    scaled_gtp = scaler_gtp.transform(gtp_feature)


    #Dental Carries and Target don't require any processing
    dental_feature = data['dental caries'].to_numpy().reshape(-1,1)
    smoking_target = data['smoking'].to_numpy().reshape(-1,1)

    # Concatenate all the transformed arrays
    concatenated_array = np.concatenate((scaled_hemoglobin, dental_feature, scaled_fasting, scaled_LDL, scaled_height, scaled_weight, scaled_serum, scaled_gtp, smoking_target), axis=1)

    # Convert the concatenated array into a DataFrame
    columns = ['hemoglobin','dental caries','fasting blood sugar','LDL','height(cm)','weight(kg)','serum creatinine','Gtp','smoking']
    concatenated_df = pd.DataFrame(concatenated_array, columns=columns)

    # Display or use the concatenated DataFrame
    return concatenated_df


def get_normalized_train():
    return get_normalized_data(pd.read_csv("train_data.csv"))


def get_normalized_test():
    return get_normalized_data(pd.read_csv("test_data.csv"))

def get_normalized_val():
    return get_normalized_data(pd.read_csv("val_data.csv"))




# #read data
# test = pd.read_csv("test_data.csv")
# train = pd.read_csv("train_data.csv")
# val = pd.read_csv("val_data.csv")

# #define features
# hemoglobin = train['hemoglobin'].to_numpy()
# dental = train['dental caries'].to_numpy()
# fasting = train['fasting blood sugar'].to_numpy()
# LDL = train['LDL'].to_numpy()
# height = train['height(cm)'].to_numpy()
# weight = train['weight(kg)'].to_numpy()
# serum = train['serum creatinine'].to_numpy()
# gtp = train['Gtp'].to_numpy()

# hemoglobin_test = test['hemoglobin'].to_numpy()
# dental_test = test['dental caries'].to_numpy()
# fasting_test = test['fasting blood sugar'].to_numpy()
# LDL_test = test['LDL'].to_numpy()
# height_test = test['height(cm)'].to_numpy()
# weight_test = test['weight(kg)'].to_numpy()
# serum_test = test['serum creatinine'].to_numpy()
# gtp_test = test['Gtp'].to_numpy()

# hemoglobin_val = val['hemoglobin'].to_numpy()
# dental_val = val['dental caries'].to_numpy()
# fasting_val = val['fasting blood sugar'].to_numpy()
# LDL_val = val['LDL'].to_numpy()
# height_val = val['height(cm)'].to_numpy()
# weight_val = val['weight(kg)'].to_numpy()
# serum_val = val['serum creatinine'].to_numpy()
# gtp_val = val['Gtp'].to_numpy()


# features = [hemoglobin, fasting, LDL, height, weight, serum, gtp]




# def get_normalized(features):
#     sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[0])
#     #scaling the data using standard scaler
#     scaler = StandardScaler()
#     scaler.fit(box_cox_transformed_data)
#     scaled_hemoglobin_train = scaler.transform(box_cox_transformed_data)


#     sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[1])
#     #scaling the data using standard scaler
#     scaler = StandardScaler()
#     scaler.fit(box_cox_transformed_data)
#     scaled_fasting_blood_sugar_train = scaler.transform(box_cox_transformed_data)

#     #scaling the data using standard scaler usnig original data
#     scaler = StandardScaler()
#     LDL_feature = features[2].reshape(-1, 1)
#     scaler.fit(LDL_feature)
#     scaled_LDL_train = scaler.transform(LDL_feature)

#     #scaling the data using min-max scaler using original data
#     scaler = MinMaxScaler()
#     height_feature = features[3].reshape(-1, 1)
#     scaler.fit(height_feature)
#     scaled_height_train = scaler.transform(height_feature)

#     #scaling the data using robust scaler beacause it is robust to outliers and can handle them
#     scaler = RobustScaler()
#     weight_feature = features[4].reshape(-1, 1)
#     scaler.fit(weight_feature)
#     scaled_weight = scaler.transform(weight_feature)

#     sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[5])
#     log_transformed_data = log_transformed_data.reshape(-1, 1)
#     #scaling the data using min-max scaler
#     scaler = MinMaxScaler()
#     scaler.fit(log_transformed_data)
#     scaled_serum_creatinine = scaler.transform(log_transformed_data)

#     sqaure_root_transformed_data, log_transformed_data, box_cox_transformed_data, yeo_johnson_transformed_data = get_tranformations(features[6])
#     #scaling the data using standard scaler
#     scaler = StandardScaler()
#     scaler.fit(box_cox_transformed_data)
#     scaled_Gtp = scaler.transform(box_cox_transformed_data)

#     return scaled_hemoglobin_train, scaled_fasting_blood_sugar_train, scaled_LDL_train, scaled_height_train, scaled_weight, scaled_serum_creatinine, scaled_Gtp






