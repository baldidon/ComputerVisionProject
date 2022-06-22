def main():
    parser = dict()
    parser['lr'] = 0.0005 # learning rate
    parser['wd'] = 0.0001  # weight decay
    parser['p'] = 0.5  # for data agumentation
    parser['batch_size'] = 16
    parser['hue'] = 0.3
    parser['contrast'] = 0.5
    parser['brightness'] = 0.5
    parser['max_epochs'] = 40 #first, i use 40
    parser['max_pool'] = 0
    parser['CIFAR'] = 10
    parser['optimizer'] = 'Adam'  #Adam or Adamw
    parser['data_agumentation'] = 1
    parser['plot'] = 1
    parser['data_root'] = f"data_{parser['CIFAR']}"
    parser['model_root'] = f"model_10" #or model_100 for model dict upload

    args = parser

    solver = Solver(args)
    solver.fit()

if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # with conda environment i've had issues with SSL certificates while downloading datasets
    
    main()
