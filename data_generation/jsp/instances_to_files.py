def generate_data_to_files(seed, directory, config):
    """
        Generate data and save it to the specified directory
    :param seed: seed for data generation
    :param directory: the directory for saving files
    :param config: other parameters related to data generation
    """
    n_j = config.n_j
    n_m = config.n_m
    source = config.data_source
    batch_size = config.data_size
    data_suffix = config.data_suffix

    suffix = strToSuffix(data_suffix)
    low = config.low
    high = config.high

    filename = '{}x{}{}'.format(n_j, n_m, suffix)
    np.random.seed(seed)
    random.seed(seed)

    print("-" * 25 + "Data Setting" + "-" * 25)
    print(f"seed : {seed}")
    print(f"data size : {batch_size}")
    print(f"data source: {source}")
    print(f"filename : {filename}")
    print(f"processing time : [{low},{high}]")
    print(f"mode : {data_suffix}")
    print("-" * 50)

    path = directory + filename

    if (not os.path.exists(path)) or config.cover_data_flag:
        if not os.path.exists(path):
            os.makedirs(path)

        for idx in range(batch_size):
            if source == 'SD2':
                job_length, op_pt, op_per_mch = SD2_instance_generator(config=config)

                lines_doc = matrix_to_text(job_length, op_pt, op_per_mch)

                doc = open(
                    path + '/' + filename + '_{}.fjs'.format(str.zfill(str(idx + 1), 3)),
                    'w')
                for i in range(len(lines_doc)):
                    print(lines_doc[i], file=doc)
                doc.close()
    else:
        print("the data already exists...")