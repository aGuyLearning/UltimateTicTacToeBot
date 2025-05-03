import os
import torch
import torch.nn as nn
import torch.nn.functional as F



class UltimateTTTNet(nn.Module):
    def __init__(self,):
        super(UltimateTTTNet, self).__init__()
        
        # Input: 2 binary planes (81 bits each) for X and O positions
        # Plus additional planes for meta-information
        
        # Main convolutional blocks
        self.conv1 = nn.Conv2d(6, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
       
        # Batch norms
        self.bn1 = nn.BatchNorm2d(128)

        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*9*9, 81)  # 81 possible moves

        
        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(9*9, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
    def forward(self, x, valid_moves_mask=None):
        # Input shape: (batch_size, 6, 9, 9)
        # Channel 0: X positions on all boards (81 bits)
        # Channel 1: O positions on all boards (81 bits)
        # Channel 2: Current active boards (9 bits)
        # Channel 3: Global X positions (9 bits)
        # Channel 4: Global O positions (9 bits)
        # Channel 5: Current player (1 bit, broadcast to 9x9)
        
        # Main convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2*9*9)
        p = self.policy_fc(p)
        policy = F.log_softmax(p, dim=1)

        # Apply move mask if provided
        if valid_moves_mask is not None:
            # Set logits of invalid moves to -inf
            p = p.masked_fill(~valid_moves_mask, float('-inf'))
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 9*9)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # Scalar between -1 and 1
        
        return policy, value

    def predict(self, state, valid_moves):
        """Convert game state to network input and get prediction"""
        with torch.no_grad():
            # Convert state to tensor
            x = state_to_tensor(state)
             # Create valid moves mask if not provided
                
            
            # Convert valid moves to mask
            mask = torch.zeros(81, dtype=torch.bool)
            for move in valid_moves:
                x_pos, y_pos = move
                mask[y_pos*9 + x_pos] = True
            
            # Add batch dimension
            mask = mask.unsqueeze(0)
            
            policy, value = self.forward(x, mask)
            return torch.exp(policy).cpu().numpy()[0], value.item()
        
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
    

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
def state_to_tensor(state):
    """Convert game state to network input tensor"""
    global_state_x, global_state_o, local_state_x, local_state_o, current_player, current_board, winner = state
    
    # Create input planes
    batch_size = 1
    planes = torch.zeros((batch_size, 6, 9, 9), dtype=torch.float32)
    
    # Plane 0: X positions on all boards
    for board in range(9):
        for y in range(3):
            for x in range(3):
                if local_state_x[board] & (1 << (y*3 + x)):
                    planes[0, 0, (board//3)*3 + y, (board%3)*3 + x] = 1
    
    # Plane 1: O positions on all boards
    for board in range(9):
        for y in range(3):
            for x in range(3):
                if local_state_o[board] & (1 << (y*3 + x)):
                    planes[0, 1, (board//3)*3 + y, (board%3)*3 + x] = 1
    
    # Plane 2: Current active boards
    if current_board == 9:  # Free choice
        for board in range(9):
            if not (global_state_x & (1 << board)) and not (global_state_o & (1 << board)):
                y_start = (board // 3) * 3
                x_start = (board % 3) * 3
                planes[0, 2, y_start:y_start+3, x_start:x_start+3] = 1
    else:
        y_start = (current_board // 3) * 3
        x_start = (current_board % 3) * 3
        planes[0, 2, y_start:y_start+3, x_start:x_start+3] = 1
    
    # Plane 3: Global X positions
    for board in range(9):
        if global_state_x & (1 << board):
            y_start = (board // 3) * 3
            x_start = (board % 3) * 3
            planes[0, 3, y_start:y_start+3, x_start:x_start+3] = 1
    
    # Plane 4: Global O positions
    for board in range(9):
        if global_state_o & (1 << board):
            y_start = (board // 3) * 3
            x_start = (board % 3) * 3
            planes[0, 4, y_start:y_start+3, x_start:x_start+3] = 1
    
    # Plane 5: Current player (1 for X, 0 for O)
    planes[0, 5, :, :] = 1 if current_player else 0
    
    return planes