import torch
tuple_info = (b'/tmp/fileAy5Jip', b'/torch_32412_3277714864', 25)
storage = torch.Storage._new_shared_filename(*tuple_info)
y = torch.Tensor(storage).view((5, 5))