

def compute_motion_mse(fut_motion_orig, train_dec_motion, fut_mask, weight):
    diff = fut_motion_orig - train_dec_motion
    mask = fut_mask  # all one
    diff *= mask.unsqueeze(2)
    loss_unweighted = diff.pow(2).sum()
    loss_unweighted /= diff.shape[0]
    loss = loss_unweighted * weight
    return loss, loss_unweighted


def compute_z_kld(q_z_dist, p_z_dist, agent_num, min_clip, weight):
    loss_unweighted = q_z_dist.kl(p_z_dist).sum()   # sn=1
    loss_unweighted /= agent_num
    loss_unweighted = loss_unweighted.clamp_min_(min_clip)
    loss = loss_unweighted * weight
    return loss, loss_unweighted


def compute_sample_loss(fut_motion_orig, infer_dec_motion, fut_mask, weight):
    diff = infer_dec_motion - fut_motion_orig.unsqueeze(1)  # handle different shapes
    mask = fut_mask.unsqueeze(1).unsqueeze(-1)
    diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    loss_unweighted = loss_unweighted.mean()
    loss = loss_unweighted * weight
    return loss, loss_unweighted


def compute_vae_loss(args, fut_motion_orig, train_dec_motion, infer_dec_motion, fut_mask, q_z_dist, p_z_dist):
    agent_num = len(fut_mask)  # need to check
    mse_loss, mse_loss_uw = compute_motion_mse(fut_motion_orig, train_dec_motion, fut_mask, args.mse_weight)
    kld_loss, kld_loss_uw = compute_z_kld(q_z_dist, p_z_dist, agent_num, args.kld_min_clamp, args.kld_weight)
    var_loss, var_loss_un = compute_sample_loss(fut_motion_orig, infer_dec_motion, fut_mask, args.var_weight)
    total_loss = mse_loss + kld_loss + var_loss
    loss_dict = {'mse': mse_loss, 'kld': kld_loss, 'sample': var_loss}
    loss_dict_uw = {'mse': mse_loss_uw, 'kld': kld_loss_uw, 'sample': var_loss_un}
    return total_loss, loss_dict, loss_dict_uw

