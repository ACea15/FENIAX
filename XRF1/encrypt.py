import pathlib
from cryptography.fernet import Fernet
import numpy as np

def generate_key():
    # Generate a key
    key = Fernet.generate_key()

    # Save the key to a file (safeguard this key; it's crucial for decryption)
    with open('/home/ac5015/pCloudDrive/mac/keys/xrf.key', 'wb') as key_file:
        key_file.write(key)
        
def load_key(file_key):
    # Load the key from the key file
    with open(file_key, 'rb') as key_file:
        key = key_file.read()
    return key

def encrypt_txt(file_data, folder_save, cipher_suite):

    with open(file_data, 'rb') as filei:
        filei_read = filei.read()
    filei_encrypted = cipher_suite.encrypt(filei_read)
    # Write the encrypted content to a new file
    with open(folder_save / f"{file_data.stem}.encrypted", 'wb') as encrypted_file:
        encrypted_file.write(filei_encrypted)

def encrypt_npy(file_data, folder_save, cipher_suite):

    datanpy = np.load(file_data)
    vshape = datanpy.shape
    if len(vshape) == 1:
        arraynpy = datanpy.tobytes()
    elif len(vshape) > 1:
        arraynpy = datanpy.reshape(np.prod(datanpy.shape)).tobytes()
    elif len(vshape) == 0:
        arraynpy = datanpy.tobytes()
    else:
        print(file_data)
        raise ValueError("can only encrypt arrays or matrices")
    array_encrypted = cipher_suite.encrypt(arraynpy)
    # Write the encrypted content to a new file
    with open(folder_save / f"{file_data.stem}.encrypted", 'wb') as encrypted_file:
        encrypted_file.write(array_encrypted)
    with open(folder_save / f"{file_data.stem}_shape.txt", 'w') as shape_file:
        shape_file.write(" ".join([str(i) for i in vshape]))

def encrypt_folder(folder_path, file_key):

    key = load_key(file_key)
    folder_path = pathlib.Path(folder_path)
    folder_encryptpath = folder_path.parent / f"{folder_path.name}encrypted"
    folder_encryptpath.mkdir(exist_ok=True)
    # Initialize the Fernet class with the key
    cipher_suite = Fernet(key)
    p = pathlib.Path(folder_path).glob('**/*')
    files = [x for x in p if x.is_file()]
    for file_i in files:
        if file_i.suffix == '.txt':
            encrypt_txt(file_i, folder_encryptpath, cipher_suite)
        elif file_i.suffix == '.npy':
            encrypt_npy(file_i, folder_encryptpath, cipher_suite)

encrypt_folder(folder_path='FEM', file_key='/home/ac5015/pCloudDrive/mac/keys/xrf.key')
encrypt_folder(folder_path='AERO', file_key='/home/ac5015/pCloudDrive/mac/keys/xrf.key')



def decrypt_folder(folder_name, file_key):
    ...
    
# Encrypt the file content
encrypted_content = cipher_suite.encrypt(original_content)

# Write the encrypted content to a new file
with open('encrypted_file.txt', 'wb') as encrypted_file:
    encrypted_file.write(encrypted_content)


key = load_key('/home/ac5015/pCloudDrive/mac/keys/xrf.key')    
cipher_suite = Fernet(key)
folder_encrypted = pathlib.Path("FEMencrypted")
folder_encrypted.mkdir(exist_ok=True)
p = pathlib.Path("FEM").glob('**/*')
files = [x for x in p if x.is_file()]
# Encrypt .txt and .npy files
for file_i in files:
    if file_i.suffix == '.txt':
        with open(file_i, 'rb') as filei:
            filei_read = filei.read()
        filei_encrypted = cipher_suite.encrypt(filei_read)
        # Write the encrypted content to a new file
        with open(folder_encrypted / f"{file_i.stem}.encrypted", 'wb') as encrypted_file:
            encrypted_file.write(filei_encrypted)




# Read the encrypted array file
with open(file_path, 'rb') as file:
    encrypted_content = file.read()

# Decrypt the array content
decrypted_content = cipher_suite.decrypt(encrypted_content)

# Load the decrypted content as a NumPy array with the original shape
decrypted_array = np.frombuffer(decrypted_content, dtype=dtype).reshape((n, n))

key = load_key('/home/ac5015/pCloudDrive/mac/keys/xrf.key')    
cipher_suite = Fernet(key)
folder_decrypted = pathlib.Path("FEMnew")
folder_decrypted.mkdir(exist_ok=True)
folder_encrypted = pathlib.Path("FEMencrypted")
p = folder_encrypted.glob('**/*')
files = [x for x in p if x.is_file()]
# Encrypt .txt and .npy files
for file_i in files:
    if file_i.suffix == '.encrypted':
        with open(file_i, 'rb') as filei:
            filei_read = filei.read()
        filei_encrypted = cipher_suite.decrypt(filei_read)
        # Write the encrypted content to a new file
        with open(folder_decrypted / f"{file_i.stem}.txt", 'wb') as encrypted_file:
            encrypted_file.write(filei_encrypted)
