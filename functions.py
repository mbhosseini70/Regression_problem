import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


preprocessor = joblib.load('preprocessor.pkl')

def preprocess_data(X):
    """
    Preprocesses the given dataframe X using a pre-fitted preprocessor.
    
    Parameters:
    - X : DataFrame to be processed

    Returns:
    - X_transformed: Processed DataFrame
    """
    X_transformed = preprocessor.transform(X)
    return X_transformed



def sort_addresses_with_other_at_end(addresses):
    """
    Sorts a list of addresses alphabetically, placing 'Other' at the end.
    
    Args:
    - addresses (list): List of addresses.
    
    Returns:
    - List: Sorted list of addresses with 'Other' at the end.
    """
    sorted_addresses = sorted(filter(lambda x: x != 'Other', addresses))
    sorted_addresses.append('Other')
    return sorted_addresses





columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']

addresses = ['Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb',
       'North Program Organization', 'Andisheh', 'West Ferdows Boulevard',
       'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi',
       'Shahrake Shahid Bagheri', 'Moniriyeh', 'Amirieh',
       'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden',
       'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad',
       'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht',
       'Aqdasieh', 'Railway', 'Central Janatabad',
       'East Ferdows Boulevard', 'Other', 'Sattarkhan', 'Shahryar',
       'Northern Janatabad', 'Daryan No', 'Southern Program Organization',
       'Rudhen', 'West Pars', 'Afsarieh', 'Marzdaran', 'Sadeghieh',
       'Pakdasht', 'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi',
       'Keshavarz Boulevard', 'Baghestan', 'Kahrizak', 'Qarchak',
       'Northren Jamalzadeh', 'Azarbaijan', 'Persian Gulf Martyrs Lake',
       'Beryanak', 'Heshmatieh', 'Elm-o-Sanat', 'Golestan',
       'Shahr-e-Ziba', 'Pasdaran', 'Gheitarieh', 'Gholhak', 'Heravi',
       'Hashemi', 'Dehkade Olampic', 'Republic', 'Zaferanieh',
       'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr', 'Qalandari',
       'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Haft Tir', 'Mahallati',
       'Ozgol', 'Tajrish', 'Abazar', 'Koohsar', 'Hekmat', 'Parastar',
       'Majidieh', 'Southern Chitgar', 'Karimkhan', 'Si Metri Ji',
       'Karoon', 'Northern Chitgar', 'East Pars', 'Kook', 'Air force',
       'Velenjak', 'Kamranieh', 'Komeil', 'Azadshahr', 'Amirabad',
       'Dorous', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan',
       'Yaftabad', 'Mehran', 'Nasim Shahr', 'Tenant', 'Fallah',
       'Eskandari', 'Shahrakeh Naft', 'Ajudaniye', 'Tehransar', 'Nawab',
       'Yousef Abad', 'Northern Suhrawardi', 'Hakimiyeh', 'Nezamabad',
       'Garden of Saba', 'Tarasht', 'Araj', 'Vahidieh', 'Malard',
       'Shahrake Azadi', 'Tehran Now', 'Darabad', 'Damavand', 'Atabak',
       'Sabalan', 'Waterfall', 'Ahang', 'Pishva', 'Ghoba', 'Shahedshahr',
       'Southern Suhrawardi', 'Abuzar', 'Dolatabad', 'Hor Square',
       'Taslihat', 'Robat Karim', 'Seyed Khandan', 'Shahrake Quds',
       'Chidz', 'Khavaran', 'Shoosh', 'Vahidiyeh']