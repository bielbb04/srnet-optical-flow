class Logger():
    def __init__(self, dataset, model_name, nickname, path_saving):
        day = datetime.now().strftime("%Y-%m-%d")
        self.model_name = model_name
        self.nickname = nickname
        self.dir_path = f"{path_saving}/checkpoints/{dataset}/{model_name}/{day}/{nickname}" if nickname is not None else f"{path_saving}/checkpoints/{dataset}/{model_name}/{day}"
        self.writer = SummaryWriter(self.dir_path)
        os.makedirs(f"{self.dir_path}/checkpoints", exist_ok=True)
        self.best_loss = float("inf")

    def log_loss(self, epoch, train_loss, validation_loss):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/validation", validation_loss, epoch)
        self.writer.add_scalars("Loss/comparison", {"train": train_loss, "validation": validation_loss}, epoch)

    def log_params(self, num_params):

        self.writer.add_text("Parameters", f'{self.model_name} = {num_params}')

    def save_checkpoints(self, model, epoch, validation_loss):
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict()}
        torch.save(ckpt, f"{self.dir_path}/checkpoints/last.ckpt")
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            torch.save(ckpt, f"{self.dir_path}/checkpoints/best.ckpt")

    def plot_results(self, epoch, gt, low, high):
        sampling = high.size(2) // low.size(2)
        inter = nn.functional.interpolate(low, scale_factor=sampling, mode="bicubic")
        images = torch.cat([gt, inter, high], dim=3)
        grid = make_grid(images, nrow=1)
        self.writer.add_image("Image comparison: Ref ~ Bic ~ Pred", grid, epoch)