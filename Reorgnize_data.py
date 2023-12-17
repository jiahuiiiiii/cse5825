import os
import pandas as pd
import shutil

def organize_photos(excel_path, dataset_path, output_path): 
    '''
    This function is used to reorgnize data 
    and extract the photo id with the corresponding features
    '''

    df = pd.read_csv(excel_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for _, row in df.iterrows():
        photo_id = row['id']
        category = row['price']

        category_path = os.path.join(output_path, str(category))
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if file_name.startswith(str(photo_id) + "_"):
                        src_path = os.path.join(subdir_path, file_name)
                        dst_path = os.path.join(category_path, file_name)
                        shutil.copy(src_path, dst_path)
                        break

if __name__ == "__main__":
    # Need to change to the real path
    organize_photos('/data2/bil22003/Bayesian/data/cat_data_full_42012.csv', 
                    '/data2/bil22003/Bayesian/age', 
                    '/data2/bil22003/Bayesian/price')
