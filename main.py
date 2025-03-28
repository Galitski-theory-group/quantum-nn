from anc import *
from sys import argv

def main():
    # Note: two arguments are passed by the system, a and g respectively.
    train_size=5000
    train_batch_size=64
    test_size=10000
    test_batch_size=10000

    width=512
    depth=3
    input_size=784
    output_size=10
    a=float(argv[1])
    g=float(argv[2])

    learning_rate=0.01
    momentum=0.9
    num_shots=15
    num_epochs=10
    step=10

    hp_dict={
        "train_size": train_size,
        "batch_size": train_batch_size,
        "test_size": test_size,
        "width": width,
        "depth": depth,
        "a": a,
        "g": g,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "num_shots": num_shots,
        "epochs": num_epochs,
        "step": step        
    }

    train_loader,test_loader=prep_data(train_size,train_batch_size,test_size,test_batch_size) # prepare training and test data
    net=QMLP(width,depth,input_size,output_size,a=a,g=g) # create quantum network
    net.apply(init_weights) # apply Kaiming intialization to weights
    res_dict=train(net,train_loader,test_loader,learning_rate,momentum,num_shots,num_epochs,step) # train the network
    record(hp_dict|res_dict,'res_data','a'+str(a)+'_g'+str(g)+'.json') # write the data to a file

    print(res_dict["test_accuracy"])

if __name__=='__main__':
    main()
