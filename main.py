import sys, os, argparse, random, glob, numpy as np, tensorflow as tf, keras.backend as K

from keras import optimizers, losses
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Concatenate, Dropout, GlobalAveragePooling2D, MaxPooling2D
from keras.backend.tensorflow_backend import set_session
from keras.applications.resnet import ResNet50

from utils import MyModels, MyWheightType, get_expected, get_data_generator, clean_global_variables, save_results

QUANT_WEIGHT_TYPE = MyWheightType.C1Q1
IMG_DIM = (256, 256, 1)

def count_CC_loss(y_true, y_pred):
  class_index = K.argmax(y_pred, axis = 1)
  y, idx, CC = tf.unique_with_counts(class_index)

##---------------------------------------------------#
## y and CC must have all possible positions contained in y_true
##---------------------------------------------------#
##Possible indices
  true_index = tf.range(y_pred.shape[1], dtype = y.dtype)
  y_aux = tf.concat([y, true_index], axis = -1)
##class_index values in y's head and other values in y's tail
  y, _ = tf.unique(y_aux)

##Possible indices have count = 0
  CC_aux = tf.zeros(y_pred.shape[1], dtype = CC.dtype)
  CC_aux = tf.concat([CC, CC_aux], axis = -1)
##CC initial values in CC's head and CC's tail has just count = 0 (padding)
  CC = tf.slice(CC_aux, [0], [y_pred.shape[1]])
#---------------------------------------------------#

##Puts CC in the same order of y
  CC = K.gather(CC, tf.contrib.framework.argsort(y))
  CC = K.cast(CC, K.floatx())
##Count mean
  CC = CC / K.cast(K.shape(idx), K.floatx())
#===================================================#

##Real count
  y_true_aux = K.cast(y_true, K.floatx())
  y_true_aux = K.mean(y_true, axis = 0)

  ERR = K.abs(y_true_aux - CC)
##Each sample has the counting error of its class
  ERR = tf.gather(ERR, class_index)

##Weighting cases
  lambda1 = lambda2 = 1.0

  if QUANT_WEIGHT_TYPE == MyWheightType.C2Q1:
    lambda2 = 0.5
  elif QUANT_WEIGHT_TYPE == MyWheightType.C1Q2:
    lambda1 = 0.5    

  return lambda1 * losses.categorical_crossentropy(y_true, y_pred) + lambda2 * ERR

def my_CNN2D(dims, qt_labels, dropout, quant, summary = False):
  input_layer  = Input(shape = (dims))
  
  layer = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
  layer = MaxPooling2D((2, 2), padding='same')(layer)
  layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)  
  layer = MaxPooling2D((2, 2), padding='same')(layer)  
  layer = Flatten()(layer)
  layer = Dense(128, activation='relu')(layer)
  layer = Dense(128, activation='relu')(layer)  

  name = "CNN{}".format("-CQ" if quant else "")

  if dropout:
    layer = Dropout(0.25)(layer)
    name = name + " with dropout"

  output_layer = Dense(qt_labels, activation='softmax')(layer)     

  opt = optimizers.Adam(lr=0.0001)
  model = Model(input_layer, output_layer, name = name)

  if quant:
    model.compile(loss = count_CC_loss, optimizer=opt, metrics=['acc'])  
  else:
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])    

  if summary:
    model.summary()
  
  return model

def my_ResNet50(dims, qt_labels, quant, summary = False):
  input_layer = Input(shape = dims)
  #Combine a gray-scale image to generate a 3D tensor
  input_layer = Concatenate()([input_layer, input_layer, input_layer])

  model = ResNet50(weights = "imagenet", include_top = False, input_shape = (dims[0], dims[1], 3), classes = qt_labels, input_tensor = input_layer)

  layer = GlobalAveragePooling2D()(model.output)
  layer = Dense(qt_labels, activation = "softmax")(layer)
  model = Model(model.input, layer, name = "ResNet50{}".format("-CQ" if quant else ""))

  opt = optimizers.SGD(lr=0.001)  

  if quant:
    model.compile(loss = count_CC_loss, optimizer=opt, metrics=['acc'])
  else:
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

  if summary:
    model.summary()
 
  return model

def train_model(args):
  train   = get_data_generator(os.path.join(args.source_path, "train"), args.batch_size)
  validation = get_data_generator(os.path.join(args.source_path, "validation"), args.batch_size)

  if args.model == MyModels.CONV2D:
    model = my_CNN2D(IMG_DIM, args.qt_label, dropout = False, quant = args.quant is not None, summary = False) 
  elif args.model == MyModels.RESNET50:  
    model = my_ResNet50(IMG_DIM, args.qt_label, args.quant is not None, summary = False)
  else:  
    raise Exception("Model is not implemented.")      

  early  = EarlyStopping(monitor = "val_loss", min_delta = 0.0001, patience = 20, mode = "min", restore_best_weights = True, verbose = 1)  
  weight = ModelCheckpoint(os.path.join(args.target_path, 'weights.hdf5'), mode = "min", save_best_only = True, verbose = 1)

  imgs  = glob.glob(os.path.join(args.source_path, "train", "*.png"))
  train_steps = int(np.ceil(len(imgs) / (args.batch_size)))
  imgs  = glob.glob(os.path.join(args.source_path, "validation", "*.png"))
  val_steps = int(np.ceil(len(imgs) / (args.batch_size)))

  model.fit_generator(train, 
                      epochs = 100, 
                      validation_data = validation, 
                      callbacks = [early, weight],
                      workers = 8,
                      max_queue_size = 15,
                      use_multiprocessing = False,
                      shuffle = False, 
                      verbose = 1,
                      steps_per_epoch = train_steps,
                      validation_steps = val_steps)  

def apply_model(args):
  model_name = glob.glob(os.path.join(args.target_path, "*.hdf5"))

  if len(model_name) > 0:
    model_name = model_name[0]
  else:
    raise Exception("There is no hdf5 file with model in {}".format(args.target_path))    

  if args.model == MyModels.CONV2D:
    model = my_CNN2D(IMG_DIM, args.qt_label, dropout = False, quant = args.quant is not None, summary = False) 
    model = load_model(model_name, compile = args.quant is None)
  elif args.model == MyModels.RESNET50:  
    model = my_ResNet50(IMG_DIM, args.qt_label, args.quant is not None, summary = False)
    model.load_weights(model_name)
  else:  
    raise Exception("Model is not implemented.")

  if args.quant is not None:
    model.compile(loss = count_CC_loss, optimizer="adam" if args.model == MyModels.CONV2D else "sgd", metrics=['acc'])  
  
  generator = get_data_generator(args.source_path, args.batch_size)
  imgs  = glob.glob(os.path.join(args.source_path, "*.png"))

  predicted = model.predict_generator(generator, steps = int(np.ceil(len(imgs) / (args.batch_size))))
  save_results(get_expected(args.source_path), predicted, args.eval)

# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
def options(args):
  if (os.path.isdir(args.source_path)):
    #======================SEED===============================#
    seed_value = 1522 #default

    random.seed(seed_value)
    np.random.seed(seed_value)

    tf.set_random_seed(seed_value)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(graph = tf.get_default_graph(), config = config))  
    #=========================================================#

    clean_global_variables()
    global QUANT_WEIGHT_TYPE

    if args.quant == 1:
      QUANT_WEIGHT_TYPE = MyWheightType.C1Q1
    elif args.quant == 2:
      QUANT_WEIGHT_TYPE = MyWheightType.C2Q1
    elif args.quant == 3:  
      QUANT_WEIGHT_TYPE = MyWheightType.C1Q2    

    if args.action == "train":
      train_model(args)
    elif args.action == "apply":
      apply_model(args)
  else:
    raise Exception("Source path doesn't exist")  

def main(*args):
  parser = argparse.ArgumentParser(description="Audio classification")
  
  parser.add_argument("-a", help = "Action to be executed", dest = "action", choices = ["train", "apply"], required = True)
  parser.add_argument("-s", help = "Directory to load spectrogram images", dest = "source_path", required = True)     
  parser.add_argument("-t", help = "Directory to save/load model. Default = current directory", dest = "target_path", required = False)
  parser.add_argument("-m", help = "Model index (6) CNN (18) ResNet ", dest = "model", required = True, type = int)
  parser.add_argument("-b", help = "Batch size used for training, validation and test. Default = 80", dest = "batch_size", required = False, default = 80, type = int)
  parser.add_argument("-e", help = "Quantity of epochs. Default = 100", dest = "qt_epoch", required = False, default = 100, type = int)
  parser.add_argument("-l", help = "Amount of labels", dest = "qt_label", required = True, type = int)
  parser.add_argument("-quant", help = "Add quantification loss function. (no value after parameter or 1) first weighting, (2) second weighting, (3) third weighting case.", dest = "quant", default = None, const = 1, type = int, nargs = '?', choices=range(1, 4)) # um ou nenum argumento
  parser.add_argument("-eval", help = "Generate model evaluation", dest = "eval", default = False, action="store_true")  

  parser.set_defaults(func = options)
  
  ARGS = parser.parse_args()
  ARGS.func(ARGS)  

if __name__ == '__main__':
  main(*sys.argv[1:])  
