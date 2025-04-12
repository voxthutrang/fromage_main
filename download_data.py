import pandas as pd
import numpy as np
import requests
import zlib
import os
import shelve
import magic #pip install python-magic
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import glob

headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}

def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)

def df_multiprocess(df, processes, chunk_size, func, dataset_name):
    print("Generating parts...")
    with shelve.open('%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size)) as results:
 
        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        print(int(len(df) / chunk_size), "parts.", chunk_size, "per part.", "Using", processes, "processes")
 
        pbar.desc = "Downloading"
        with Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return

# Unique name based on url
def _file_name(row):
    return "%s/%s_%s" % (row['folder'], row.name, (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row['file'])):
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
    return row

# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(row['url'], stream=False, timeout=5, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
    if response.ok:
        row['file'] = fname
    return row

def download_image(row):
    fname = _file_name(row)
    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        #row['headers'] = dict(response.headers)
    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
   
    if response.ok:
        try:
            content_type = response.headers.get("Content-Type", "")
            ext = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp"
            }.get(content_type.split(";")[0].strip(), "")

            fname = fname + ext

            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)
            row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row

def open_tsv(fname, folder, nrows):
    print(f"Opening {fname} Data File (only first {nrows} rows)...")
    os.makedirs(folder, exist_ok=True)
    if nrows is not None:
        df = pd.read_csv(fname, sep='\t', names=["caption", "url"], nrows=nrows)
    else:
        df = pd.read_csv(fname, sep='\t', names=["caption", "url"])
    df['folder'] = folder
    print(f"Processing {len(df)} Images from {fname}:")
    return df

def df_from_shelve(chunk_size, func, dataset_name, original_df):
    print("Generating Dataframe from results...")
    with shelve.open('%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size)) as results:
        keylist = sorted([int(k) for k in results.keys()])
        headers = ['file', 'folder', 'mimetype', 'size', 'status', 'url']
        df = pd.DataFrame()
        for key in keylist:
            chunk_data = results[str(key)][1]
            # Assuming chunk_data is a list of lists (rows)
            chunk_df = pd.DataFrame(chunk_data, columns=headers)
            # Filter chunk_df for rows where 'status' is 200
            filtered_chunk_df = chunk_df[chunk_df['status'] == 200]
            # Append the filtered chunk to the main DataFrame
            df = pd.concat([df, filtered_chunk_df], ignore_index=True)
        
        # merge original tsv data with downloaded data
        merged_df = pd.merge(original_df, df, on='url', how='inner')

        # only keeps status 200
        merged_df = merged_df[merged_df['status'] == 200]

        # drop redundant columns
        columns_to_keep = ['caption', 'file']
        columns_to_drop = [col for col in merged_df.columns if col not in columns_to_keep]
        merged_df = merged_df.drop(columns=columns_to_drop)

        # clean up image name
        merged_df['file'] = merged_df['file'].apply(lambda x: os.path.basename(x))

        # only use files with jpg extension
        merged_df = merged_df[merged_df['file'].str.lower().str.endswith('.jpg')]

        # rename column from file to image
        merged_df = merged_df.rename(columns={'file': 'image'})

    return merged_df

def resize_image(image_path, size=(256, 256)):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB") 
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(image_path, format="JPEG")
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")

if __name__ == "__main__":

    # number of processes in the pool can be larger than cores
    num_processes = 32
    # chunk_size is how many images per chunk per process - changing this resets progress when restarting.
    images_per_part = 100

    import sys
    train_nrows = int(sys.argv[1]) if len(sys.argv) > 2 else None
    val_nrows = int(sys.argv[2]) if len(sys.argv) > 1 else None

    data_name = "data/cc3m/validation"
    original_data = open_tsv("datasets/Validation_GCC-1.1.0-Validation.tsv", data_name, nrows=val_nrows)
    df_multiprocess(df=original_data, processes=num_processes, chunk_size=images_per_part, func=download_image, dataset_name=data_name)
    df = df_from_shelve(chunk_size=images_per_part, func=download_image, dataset_name=data_name, original_df=original_data)
    output_path = "datasets/cc3m_val.tsv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print("Saved.")
    image_paths = glob.glob(os.path.join(data_name, "*.jpg"))
    for path in image_paths:
        resize_image(path)
    print("Resized")

    data_name = "data/cc3m/training"
    original_data = open_tsv("datasets/Train_GCC-training.tsv", data_name, nrows=train_nrows)
    df_multiprocess(df=original_data, processes=num_processes, chunk_size=images_per_part, func=download_image, dataset_name=data_name)
    df = df_from_shelve(chunk_size=images_per_part, func=download_image, dataset_name=data_name, original_df=original_data)
    output_path = "datasets/cc3m_train.tsv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print("Saved.")
    image_paths = glob.glob(os.path.join(data_name, "*.jpg"))
    for path in image_paths:
        resize_image(path)
    print("Resized")