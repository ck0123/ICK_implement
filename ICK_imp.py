import torch
import main
def do_ICK(now_epoch):

    checkpoint = torch.load('./drive/My Drive/checkpoint/ckpt_resnet18_' + str(now_epoch) + '.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    ct = 0
    for child in net.module.children():
        ct += 1
        if ct < len(list(net.module.children())):
            for param in child.parameters():
                param.requires_grad = False
        else:
            child.reset_parameters()

    print("freeze the model Done.")

    # run the fine-tune with frozen model
    for epoch in range(start_epoch + 1, start_epoch + 11):
        train(net, epoch)
        if (epoch == start_epoch + 10):
            test(net, epoch, 'resnet18', True, "_frozen")