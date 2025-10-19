def warp_image(img: Tensor,
         flow: Tensor,
         padding_mode = "border",
         interpolation: str = "bilinear",
         align_corners: bool = False) -> Tensor:

    device = img.device
    if len(flow.shape) == 3:
        flow = flow.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    b, c, h, w = img.shape

    xs = torch.arange(w, dtype=torch.float, device=flow.device).view(
        1, 1, 1, -1).expand(b, 1, h, w)
    ys = torch.arange(h, dtype=torch.float, device=flow.device).view(
        1, 1, -1, 1).expand(b, 1, h, w)
    grid = torch.cat((xs, ys), dim=1).permute(0, 2, 3, 1)

    #print("Average flow:", torch.mean(flow), flow.shape)
    if flow.shape[-1] != 2:
        grid += flow.permute(0, 2, 3, 1)
    else:
        grid += flow

    #print(grid.shape)
    grid[:,:,:,0] = 2.0 * grid[:,:,:,0] / (w - 1) - 1.0
    grid[:,:,:,1] = 2.0 * grid[:,:,:,1] / (h - 1) - 1.0
    grid = grid.to(device)
    warped_img = F.grid_sample(img, grid, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return warped_img

def read_image(img_path):
    img_pil = Image.open(img_path)
    return TF.pil_to_tensor(img_pil).to(torch.float)/255.

def generate_random_shift(img):
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
        b, c, h, w = img.shape
        flow_fake = torch.zeros(b, 2, h, w)

        flow_fake[:, 0, :, :] = uniform(0, 5)
        flow_fake[:, 1, :, :] = uniform(0, 5)

        # Aplicar warp
        I_1 = warp_image(img, flow_fake, padding_mode='zeros')
        I_1 = I_1.detach()

        return I_1, flow_fake

def gradient(u, stack=False):
    # Input should have 4 dimensions: [B, C, H, W]
    if len(u.shape) == 3:
        u = u.unsqueeze(0)
    if len(u.shape) == 2:
        u = u.unsqueeze(0).unsqueeze(0)

    dx = torch.zeros_like(u)
    dy = torch.zeros_like(u)
    dx[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
    dy[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
    if stack:
        return torch.stack((dx, dy), dim=-1)
    else:
        return dx, dy

def warp_and_derivative(img, flow):
    grad = gradient(img, stack=True)
    grad_im1_1 = grad[..., 0]
    grad_im1_2 = grad[..., 1]

    grad_warp_im1_1 = warp_image(grad_im1_1, flow)
    grad_warp_im1_2 = warp_image(grad_im1_2, flow)
    # warp_I_1_grad = warp_image(grad, flow)
    warp_I_1 = warp_image(img, flow)
    #print('warp_I_1 shape:', warp_I_1.shape, 'grad_warp_im1_1 shape:', grad_warp_im1_1.shape, 'grad_warp_im1_2 shape:', grad_warp_im1_2.shape)
    warp_I_1_grad = torch.cat([grad_warp_im1_1, grad_warp_im1_2], dim=1) # Modified
    #print('warp_I_1_grad shape:', warp_I_1_grad.shape)
    return warp_I_1, warp_I_1_grad


def rho(I_0, warp_I_1, warp_I_1_grad, u, u_0):
    #print('shape I_0:', I_0.shape, 'shape warp_I_1:', warp_I_1.shape, 'shape warp_I_1_grad:', warp_I_1_grad.shape, 'shape u:', u.shape, 'shape u_0:', u_0.shape)
    prod_esc = warp_I_1_grad * (u - u_0)

    prod_esc = torch.sum(prod_esc, dim=1, keepdim=True)  # Modified to keep the channel dimension
    #print('prod_esc shape:', prod_esc.shape, 'warp_I_1 shape:', warp_I_1.shape, 'I_0 shape:', I_0.shape)
    rho = prod_esc + warp_I_1 - I_0

    return rho


def rho_derivative(I_0, warp_I_1, warp_I_1_grad,  flow_u_k, flow_init):
    rho_flow = rho(I_0, warp_I_1, warp_I_1_grad, flow_u_k, flow_init)

    diff_flow = flow_u_k - flow_init
    # We compute the gradient of rho
    #print('rho_flow shape:', rho_flow.shape, 'warp_I_1_grad shape:', warp_I_1_grad.shape, 'warp_I_1_grad_0 shape:', warp_I_1_grad[:, 0, ...].shape)
    gradient_rho_1 = rho_flow * warp_I_1_grad[:, 0, ...].unsqueeze(1) #* diff_flow[..., 0]
    gradient_rho_2 = rho_flow * warp_I_1_grad[:, 1, ...].unsqueeze(1) #* diff_flow[..., 1]
    #print('gradient_rho_1 shape:', gradient_rho_1.shape, 'gradient_rho_2 shape:', gradient_rho_2.shape)
    gradient_rho = torch.cat([gradient_rho_1, gradient_rho_2], dim=1)

    return rho_flow, gradient_rho