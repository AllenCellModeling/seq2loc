import torch
import pdb

def weights_init(m, meth='normal'):
    
#     if meth == 'normal':

#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             try:
#                 m.weight.data.normal_(0.0, 0.02)
#             except:
#                 pass
#         elif classname.find('BatchNorm') != -1:
#             m.weight.data.normal_(1.0, 0.02)
#             m.bias.data.fill_(0)
            
#     elif meth == 'xavier_normal'
    try:
        torch.nn.init.xavier_normal_(m.weight.data, gain=1)
        torch.nn.init.xavier_normal_(m.bias.data, gain=1)
    except:
        pass


def load_state(model, optimizer, path, gpu_id = 0):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model'])
    model.cuda(gpu_id)

    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)
            
def save_state(model, optimizer, path, gpu_id = 0):
    
    model.cpu()
    optimizer.state = set_gpu_recursive(optimizer.state, -1)
    
    checkpoint = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict()}
    
    torch.save(checkpoint, path)
    
    model.cuda(gpu_id)
    optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)
    
    
def set_gpu_recursive(var, gpu_id = 0):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except:
                pass
    return var  

