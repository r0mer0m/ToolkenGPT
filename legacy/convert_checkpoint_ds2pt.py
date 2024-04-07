from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

if __name__ == '__main__':
    # ds_best_model_path = './checkpoints/emb-exp-funcqa-aug-args-loss-call-only/funcqa/epoch=7-step=4888.ckpt/'
    # pt_best_model_path = './checkpoints/emb-exp-funcqa-aug-args-loss-call-only/funcqa/best_model.pt'

    # ds_best_model_path = './checkpoints/emb-exp-funcqa-aug-args-loss-call-only-rm-equal/funcqa/epoch=7-step=4888.ckpt/'
    # pt_best_model_path = './checkpoints/emb-exp-funcqa-aug-args-loss-call-only-rm-equal/funcqa/best_model.pt'

    convert_zero_checkpoint_to_fp32_state_dict(
                ds_best_model_path, pt_best_model_path)