import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# do reconstruction
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(config.num_topics, config.vocab_size)
        self.fc.apply(self.xavier_weight_init)
        self.bn = nn.BatchNorm1d(config.vocab_size, affine=False)

    def xavier_weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, z):
        """
        Args:
            z: latent representation, shape (batch_size, num_topics)
        Returns:
            x_recon: Reconstructed documents, shape (batch_size, vocab_size)
        """
        x_recon = F.softmax(self.bn(self.fc(z)), dim=1)
        return x_recon
    