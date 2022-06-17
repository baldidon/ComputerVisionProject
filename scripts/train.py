from progetto_cv.scripts.solver import Solver
def main():
    parser = dict()
    parser['lr'] = 0.0001
    parser['wd'] = 0.0001
    parser['p'] = 0.5
    parser['batch_size'] = 16
    parser['max_epochs'] = 50 #first, i use 50
    parser['CIFAR'] = 10
    parser['data_agumentation'] = 1
    parser['optimizer'] = 'Adam'
    parser['plot'] = 1
    parser['dropout'] = 0.6
    parser['print_every'] = 1
    parser['data_root'] = f"data_{parser['CIFAR']}"
    parser['model_root'] = f"model_10" #or model_100
    parser['transfer_learning'] = False
    # FUNZIONA COSÌ, SE CIFAR VALE 100  E TRANSFER_LEARNING = True, valuto prestazioni modello
    # usando rete addestrata su CIFAR 10

    args = parser

    solver = Solver(args)
    solver.fit()

if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
