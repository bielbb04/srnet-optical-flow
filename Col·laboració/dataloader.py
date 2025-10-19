class Patches_translation(Dataset):
    def __init__( self, path, subset, sampling, patch_size = None):
        super().__init__()
        if patch_size is not None:
            if 200 % patch_size != 0:
                raise ValueError('Patch size should be a divisor of 200')
        if sampling not in [2,4,8]:
            raise ValueError('Sampling factor should be 2, 4 or 8')
        img_path_list = sorted([f'{path}/{subset}/{img}' for img in os.listdir(f'{path}/{subset}') if img.endswith('.png')])

        self.sampling = sampling
        self.patch_size = patch_size
        self.subset = subset
        self.gt, self.shifted, self.flow = self.generate_data(img_path_list)

        #print(f"Shape de gt: {self.gt.shape}")
        #print(f"Shape de shifted: {self.shifted.shape}")

    def __getitem__(self, index) :
        return self.gt[index, ...], self.shifted[index, ...], self.flow[index, ...]

    def __len__(self):
        return len(self.gt)


    def generate_data(self, img_path_list):
        gt_list = []
        shifted_list = []
        flow_list = []

        for i,img_path in enumerate(img_path_list):
            #print(f"\nProcessant imatge {i+1}/{len(img_path_list)}: {img_path}")
            gt = read_image(img_path)
            #print(f"Shape de gt: {gt.shape}")
            if self.patch_size is not None:
                gt = gt.unsqueeze(0)

                gt = nn.functional.unfold(gt, kernel_size=self.patch_size, stride=self.patch_size)

                #print(f'Shape despres de unfold: {gt.shape}')
                gt_fold = nn.functional.fold(gt, 200, kernel_size=self.patch_size, stride=self.patch_size).squeeze(0)

                gt = rearrange(gt, 'b (c p1 p2) n -> (b n) c p1 p2', p1=self.patch_size, p2=self.patch_size)
                #print(f"Shape despres de rearrange: {gt.shape}")

            # 128 parelles
            for _ in range(16):
                gt_list.append(gt)
                shifted_small_list = []
                flow_small_list = []
                for idx in range(len(gt)):
                    shifted_i, flow_i = generate_random_shift(gt[idx])
                    shifted_small_list.append(shifted_i)
                    flow_small_list.append(flow_i)
                shifted = torch.cat(shifted_small_list, dim=0)
                flow = torch.cat(flow_small_list, dim = 0)
                flow_list.append(flow)
                if len(shifted.shape) == 3:
                    shifted = shifted.unsqueeze(0)
                if self.patch_size is not None:
                    shifted = nn.functional.unfold(shifted, kernel_size=self.patch_size, stride=self.patch_size)
                    shifted = rearrange(shifted, 'b (c p1 p2) n -> (b n) c p1 p2', p1=self.patch_size, p2=self.patch_size)
                shifted_list.append(shifted)

        return torch.cat(gt_list, dim=0), torch.cat(shifted_list, dim=0), torch.cat(flow_list, dim = 0)