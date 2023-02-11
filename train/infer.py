import torch


def get_latent_representation(model, train_dataloader, config):
    """
    Args:
        model: torch.nn.Module object
        train_dataloader: torch.utils.data.DataLoader object
        config: instance of Config class
    Returns:
        latent_representation_vectors: latent representation for the input, shape (num_samples, num_topics)
    """
    model.eval()
    latent_representation_vectors = {} 
    for i, (x, tfidf_batch) in enumerate(train_dataloader):
        with torch.no_grad():
            mean_prior, var_prior, log_var_prior, \
            mean_pos, var_pos, log_var_pos, x_recon, z, z_neg, z_pos = model(x, tfidf_batch)
            ids = torch.arange(config.batch_size) + i * config.batch_size
            latent_representation_vectors.update(dict(zip(ids, z)))
    latent_vectors_list = list(latent_representation_vectors.values())
    latent_vectors_list = list(map(lambda x: x.reshape(1, config.num_topics), latent_vectors_list))
    latent_vectors = torch.cat(latent_vectors_list, dim=0)
    return latent_vectors
