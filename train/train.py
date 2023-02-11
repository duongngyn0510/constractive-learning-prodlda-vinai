import torch


def compute_beta(z, z_pos, z_neg):
    """
    Args:
        z: News representation, shape (batch_size, num_topics)
        z_neg: Negative representation, shape (batch_size, num_topics)
        z_pos: Positive representation, shape (batch_size, num_topics)
    Returns:
        beta: Tensor.float
    """
    positive_product = torch.einsum('b n, b n -> b', z, z_pos)    # z . z+
    negative_product = torch.einsum('b n, b n -> b', z, z_neg)    # z . z-
    gamma = positive_product / negative_product
    beta = gamma.mean()
    return beta


def loss(config, x, mean_prior, var_prior, log_var_prior, 
         mean_pos, var_pos, log_var_pos, x_recon, 
         z, z_neg, z_pos, beta):
    """
    Args:
        config: instance of Config class
        x: bag-of-word input, shape (batch_size, vocab_size)
        mean_prior: shape (1, num_topics)
        var_prior: shape (1, num_topics)
        log_var_prior: shape (1, num_topics) 
        mean_pos: shape (batch_size, num_topics)
        var_pos: shape (batch_size, num_topics)
        log_var_pos: shape (batch_size, num_topics)
        x_recon: Reconstructed documents, shape (batch_size, vocab_size)
        z: News representation, shape (batch_size, num_topics)
        z_neg: Negative representation, shape (batch_size, num_topics) 
        z_pos: Positive representation, shape (batch_size, num_topics) 
        beta: float, coefficent for negative samples term
    Returns:
        loss: Tensor.float
    """
    # NL
    NL = -(x * (x_recon + 1e-10).log()).sum(dim=1)
    
    # KLD
    var_division = var_pos / var_prior
    diff = mean_pos - mean_prior
    diff_term = (diff * diff) / var_prior
    logvar_division = log_var_prior - log_var_pos
    KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - config.num_topics)
    
    # Constractive
    if config.train_cl:
        positive_product = torch.einsum('b n, b n -> b', z, z_pos).exp()    # z . z+
        negative_product = torch.einsum('b n, b n -> b', z, z_neg).exp()    # z . z-
        CL = - (positive_product / (positive_product + beta * negative_product)).log()
        loss = NL + KLD + CL
    else:
        loss = NL + KLD
    return loss.mean() 
    

def train(model, train_dataloader, config, optimizer):
    """
    Args:
        model: torch.nn.Module object
        train_dataloader: torch.utils.data.DataLoader object
        config: instance of Config class
        optimizer: torch.optim object
    Returns:
        None
    """
    T = len(train_dataloader)
    model.train()
    ids = (torch.arange(config.batch_size) * config.vocab_size).unsqueeze(-1)
    ids = ids.to(config.device)
    for i, (x, tfidf_batch) in enumerate(train_dataloader):
        model.zero_grad()
        mean_prior, var_prior, log_var_prior, \
        mean_pos, var_pos, log_var_pos, x_recon, z, z_neg, z_pos = model(x, tfidf_batch, ids)
        # get beta
        if config.train_cl:
            if i == 0:
                beta_0 = compute_beta(z.detach(), z_neg.detach(), z_pos.detach())
                beta = beta_0
            else:
                beta = (1 / 2) - (1 / T) * abs(T / 2 - i) + beta
        else:
            beta = 0
        # compute loss
        cs_loss = loss(config, x, mean_prior, var_prior, log_var_prior,
                        mean_pos, var_pos, log_var_pos, x_recon,
                        z, z_neg, z_pos, beta)
        cs_loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"loss: {cs_loss:>7f}  [{i:>5d}/{T:>5d}]")


