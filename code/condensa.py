# Construct pre-trained model
criterion = torch.nn.CrossEntropyLoss()
train(model, num_epochs, trainloader, criterion)

# Instantiate compression scheme
prune = condensa.schemes.FilterPrune()
# Define objective function
tput = condensa.objectives.throughput
# Specify optimization operator
obj = condensa.searchops.Maximize(tput)
# Instantiate L-C optimizer
lc = condensa.optimizers.LC(steps=30, lr=0.01)
# Build model compressor instance
compressor = condensa.Compressor(
    model=model, # Trained model
    objective=obj, # Objective
    eps=0.02, # Accuracy threshold
    optimizer=lc, # Accuracy recovery
    scheme=prune, # Compression scheme
    trainloader=trainloader, # Train dataloader
    testloader=testloader, # Test dataloader
    valloader=valloader, # Val dataloader
    criterion=criterion # Loss criterion
)
# Obtain compressed model
wc = compressor.run()
