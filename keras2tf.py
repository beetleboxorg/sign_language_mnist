def keras2tf(keras_json,keras_hdf5,tfckpt,tf_graph):
        
    # set learning phase for no training
    backend.set_learning_phase(0)

    # if name of JSON file provided as command line argument, load from 
    # arg.keras_json and args.keras_hdf5.
    # if JSON not provided, assume complete model is in HDF5 format
    if (keras_json != ''):
        json_file = open(keras_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(keras_hdf5)

    else:
        loaded_model = load_model(keras_hdf5)


    print ('Keras model information:')
    print (' Input names :',loaded_model.inputs)
    print (' Output names:',loaded_model.outputs)
    print('-------------------------------------')

    # set up tensorflow saver object
    saver = tf.train.Saver()

    # fetch the tensorflow session using the Keras backend
    tf_session = backend.get_session()

    # get the tensorflow session graph
    input_graph_def = tf_session.graph.as_graph_def()

    # get the TensorFlow graph path, flilename and file extension
    tfgraph_path = os.path.dirname(tf_graph)
    tfgraph_filename = os.path.basename(tf_graph)
    _, ext = os.path.splitext(tfgraph_filename)

    if ext == '.pbtxt':
        asText = True
    else:
        asText = False

    # write out tensorflow checkpoint & inference graph for use with freeze_graph script
    save_path = saver.save(tf_session, tfckpt)
    tf.train.write_graph(input_graph_def, tfgraph_path, tfgraph_filename, as_text=asText)

    print ('TensorFlow information:')
    print (' Checkpoint saved as:',tfckpt)
    print (' Graph saved as     :',os.path.join(tfgraph_path,tfgraph_filename))
    print('-------------------------------------')

    return