import os, numpy as np, glob, pandas as pd, threading
from PIL import Image
from enum import IntEnum
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, balanced_accuracy_score

class MyWheightType(IntEnum): 
  C1Q1 = 0
  C2Q1 = 1
  C1Q2 = 2

class MyModels(IntEnum):
  CONV2D   = 6
  RESNET50 = 18

LOCK = None
IMGS_PREDICTED = []
LABEL_ENCODER  = None
ONEHOT_ENCODER = None

def clean_global_variables():
  global IMGS_PREDICTED, LABEL_ENCODER, ONEHOT_ENCODER, LOCK
  IMGS_PREDICTED = []
  LABEL_ENCODER  = None
  ONEHOT_ENCODER = None
  LOCK = threading.Lock()

def create_encoder(base):
  global LABEL_ENCODER, ONEHOT_ENCODER

  if LABEL_ENCODER is None or ONEHOT_ENCODER is None:
    LABEL_ENCODER = LabelEncoder()
    ONEHOT_ENCODER = OneHotEncoder(sparse = False, categories = "auto")

    data = LABEL_ENCODER.fit_transform(base)  
    data = data.reshape(len(data), 1)

    ONEHOT_ENCODER.fit_transform(data)  

def encode_labels(labels, base):
  global LABEL_ENCODER, ONEHOT_ENCODER

  create_encoder(base)
  code = LABEL_ENCODER.transform(labels) 
  code = code.reshape(len(code), 1)
  code = ONEHOT_ENCODER.transform(code)

  return code[0] 

def decode_labels(codes):
  global LABEL_ENCODER, ONEHOT_ENCODER

  if LABEL_ENCODER is None or ONEHOT_ENCODER is None:
    # #Hard coded for all 12-class
    base = ["aden_marm", "apla_leuc", "basi_culi", "boan_albo", "cycl_guja", "dend_minu", "isch_guen", "myio_leuc", "phys_cuvi", "pita_sulp", "vire_chiv", "zono_cape"]
    # #Hard coded for all bird-class
    # base = ["basi_culi", "cycl_guja", "myio_leuc", "pita_sulp", "vire_chiv", "zono_cape"] 
    # #Hard coded for all anuran-class
    # base = ["aden_marm", "apla_leuc", "boan_albo", "dend_minu", "isch_guen", "phys_cuvi"]
    # #Hard coded for all 2-class
    # base = ["vire_chiv", "phys_cuvi"]
    create_encoder(base)

  codes  = ONEHOT_ENCODER.inverse_transform(codes)

  return LABEL_ENCODER.inverse_transform(codes.ravel())

def save_ids(id):
  LOCK.acquire()

  try:
    IMGS_PREDICTED.append(id)  
  finally:
    LOCK.release()        

def load_images(files, file_label):
  imgs = []
  labels = []

  for f in files:
    save_ids(os.path.basename(f))
    img = Image.open(f)
    img = np.asarray(img)

    if len(img.shape) == 3:
      img = img[:, :, 0] #just first channel

    img = img.astype(np.float32) / 255.0 #never forget this line
    imgs.append(img)  

    label = file_label.loc[os.path.basename(f)]["label"]
    code = encode_labels([label], file_label["label"])
    labels.extend([code])

  imgs = np.array(imgs)
  imgs = np.expand_dims(imgs, 3) #images with the needed shape

  return imgs, np.array(labels)

def get_expected(path):
  labels = glob.glob(os.path.join(path, "*.csv"))

  if len(labels) > 0:
    return pd.read_csv(labels[0]) 
  else:
    raise Exception("There is no csv file with labels in {}".format(path))  

def get_data_generator(path, batch_size):
  imgs  = glob.glob(os.path.join(path, "*.png"))
  labels = glob.glob(os.path.join(path, "*.csv"))

  if len(labels) > 0:
    labels = pd.read_csv(labels[0]) 
    labels.set_index("file", inplace = True)
  else:
    raise Exception("There is no csv file with labels in {}".format(path))    

  while True:
    start = 0
    end   = batch_size

    while start < len(imgs):
      yield load_images(imgs[start:end], labels)

      start += batch_size
      end   += batch_size

def save_results(expected, predicted_codes, evaluate):
  files = []
  label_code = []

  for i, p in enumerate(predicted_codes):
    files.append(IMGS_PREDICTED[i])
    aux   = np.zeros(len(p))
    aux[np.argmax(p)] = 1
    label_code.append(aux)

  predicted = decode_labels(label_code)

  data = {"file": files, "predicted": predicted}
  data = pd.DataFrame(data)
  data.to_csv("predicted_labels.csv", index = False)

  if evaluate:
    expected = expected.sort_values(by = "file", ascending = False)["label"].to_numpy()
    predicted = data.sort_values(by = "file", ascending = False)["predicted"].to_numpy()

    file = open("model_metrics.txt", "w")

    try:
      file.write(classification_report(expected, predicted, zero_division = 0)) 
      file.write("\n\n")     
      file.write("Balanced accuracy: {:.2f}\n".format(balanced_accuracy_score(expected, predicted)))
    finally:
      file.close()          
