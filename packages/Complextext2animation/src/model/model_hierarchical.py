import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from nlp.lstm import BERTSentenceEncoder
import pdb
import pickle as pkl
import numpy as np


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class Integrator(nn.Module):
    '''
    A velocity integrator.
    If we have displacement values for translation and such, and we know the exact timesteps of the signal, 
    we can calculate the global values efficiently using a convolutional layer with weights set to 1 and kernel_size=timesteps

    Note: this method will not work for realtime scenarios. Although, it is efficient enough to keep adding displacements over time
    '''

    def __init__(self, channels, time_steps):
        super(Integrator, self).__init__()
        self.conv = CausalConv1d(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=time_steps,
                                 stride=1,
                                 dilation=1,
                                 groups=channels,
                                 bias=False)
        self.conv.weight = nn.Parameter(torch.ones_like(
            self.conv.weight), requires_grad=False)

    def forward(self, xs):
        return self.conv(xs)


class TeacherForcing():
    '''
    Sends True at the start of training, i.e. Use teacher forcing maybe.
    Progressively becomes False by the end of training, start using gt to train
    '''

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def __call__(self, epoch, batch_size=1):
        p = epoch*1./self.max_epoch
        random = torch.rand(batch_size)
        return (p < random).double()

# Sequence to Sequence AutoEncoder


class TrajEncoder(nn.Module):
    def __init__(self, hidden, num_layers=1):
        super(TrajEncoder, self).__init__()
        # self.traj = nn.Linear(h1, h2)
        self.rnn = nn.GRU(1, hidden, num_layers=num_layers, batch_first=True)

    def forward(self, traj_input):
        traj_layer1, h = self.rnn(traj_input.unsqueeze(-1))
        # output_traj = self.traj(traj_layer1)
        return traj_layer1[:, -1, :]


class TrajDecoder(nn.Module):
    def __init__(self, hidden, num_layers=1):
        super(TrajDecoder, self).__init__()
        self.traj = nn.Linear(hidden, 1)
        self.rnn = nn.GRUCell(1, hidden)
        self.tf = TeacherForcing(0.1)

    def forward(self, traj_z, timesteps, traj_input, epoch=np.inf):
        traj = traj_input[:, 0]
        traj = traj.unsqueeze(-1)
        Y = []
        h = traj_z
        for t in range(timesteps):
            # hidden = torch.cat((traj_z[:,t,:], h), dim=-1)
            h = self.rnn(traj, h)
            traj = self.traj(h) + traj
            Y.append(traj.unsqueeze(1))
            if t > 0:
                mask = self.tf(epoch, h.shape[0]).double(
                ).view(-1, 1).to(traj.device)
                traj = mask * traj_input[:, t-1] + (1-mask) * traj

        return torch.cat(Y, dim=1)


class PoseEncoder(nn.Module):
    def __init__(self, h1, h2, h3, h4, num_layers=1):
        super(PoseEncoder, self).__init__()
        self.layer1_rarm_enc = nn.Linear(9, h1)
        self.layer1_larm_enc = nn.Linear(9, h1)
        self.layer1_rleg_enc = nn.Linear(15, h1)
        self.layer1_lleg_enc = nn.Linear(15, h1)
        self.layer1_torso_enc = nn.Linear(16, h1)
        self.layer2_rarm_enc = nn.Linear(2*h1, h2)
        self.layer2_larm_enc = nn.Linear(2*h1, h2)
        self.layer2_rleg_enc = nn.Linear(2*h1, h2)
        self.layer2_lleg_enc = nn.Linear(2*h1, h2)
        self.layer3_arm = nn.Linear(2*h2, h3)
        self.layer3_leg = nn.Linear(2*h2, h3)
        self.batchnorm_full = nn.BatchNorm1d(2*h3)
        self.layer4_enc = nn.GRU(
            2*h3, h4, num_layers=num_layers, batch_first=True)

    def forward(self, P_in):
        # poseinput is of shape [b,t,63]
        right_arm = P_in[..., 22:31]
        left_arm = P_in[..., 13:22]
        right_leg = P_in[..., 46:61]
        left_leg = P_in[..., 31:46]
        mid_body = torch.cat((P_in[..., :13], P_in[..., 61:64]), dim=-1)

        right_arm_layer1 = self.layer1_rarm_enc(right_arm)
        left_arm_layer1 = self.layer1_larm_enc(left_arm)
        mid_body_layer1 = self.layer1_torso_enc(mid_body)
        right_leg_layer1 = self.layer1_rleg_enc(right_leg)
        left_leg_layer1 = self.layer1_lleg_enc(left_leg)

        right_arm_layer2 = self.layer2_rarm_enc(
            torch.cat((right_arm_layer1, mid_body_layer1), dim=-1))
        left_arm_layer2 = self.layer2_larm_enc(
            torch.cat((left_arm_layer1, mid_body_layer1), dim=-1))
        right_leg_layer2 = self.layer2_rleg_enc(
            torch.cat((right_leg_layer1, mid_body_layer1), dim=-1))
        left_leg_layer2 = self.layer2_lleg_enc(
            torch.cat((left_leg_layer1, mid_body_layer1), dim=-1))

        upperbody = self.layer3_arm(
            torch.cat((right_arm_layer2, left_arm_layer2), dim=-1))
        lower_body = self.layer3_leg(
            torch.cat((right_leg_layer2, left_leg_layer2), dim=-1))

        full_body = torch.cat((upperbody, lower_body), dim=-1)
        fullbody_bn = self.batchnorm_full(
            full_body.permute(0, 2, 1)).permute(0, 2, 1)
        z_p, h = self.layer4_enc(fullbody_bn.view(
            full_body.shape[0], full_body.shape[1], -1))
        return z_p[:, -1, :]


class VelDecoder(nn.Module):
    def __init__(self, h1, h2, h3, h4):
        super(VelDecoder, self).__init__()
        self.rnn = nn.GRUCell(64, h4)
        self.cell = VelDecoderCell(h1, h2, h3, h4)
        # Hardcoded to reach 0% Teacher forcing in 10 epochs
        self.tf = TeacherForcing(0.1)

    def forward(self, h, time_steps, gt, epoch=np.inf):

        x = gt[:, 0, :]  # starting point for the decoding

        Y = []
        for t in range(time_steps):
            h = self.rnn(x, h)
            x = self.cell(h) + x
            Y.append(x.unsqueeze(1))
            if t > 0:
                mask = self.tf(epoch, h.shape[0]).double(
                ).view(-1, 1).to(x.device)
                x = mask * gt[:, t-1, :] + (1-mask) * x
        return torch.cat(Y, dim=1)

    def sample(self, h, time_steps, start):
        x = start  # starting point for the decoding
        Y = []
        for t in range(time_steps):
            h = self.rnn(x, h)
            x = self.cell(h) + x
            Y.append(x.unsqueeze(1))
        return torch.cat(Y, dim=1)


class VelDecoderCell(nn.Module):
    def __init__(self, h1, h2, h3, h4):
        super(VelDecoderCell, self).__init__()

        self.layer1_rarm_dec = nn.Linear(h1, 9)
        self.layer1_larm_dec = nn.Linear(h1, 9)
        self.layer1_rleg_dec = nn.Linear(h1, 15)
        self.layer1_lleg_dec = nn.Linear(h1, 15)
        self.layer1_torso_dec = nn.Linear(h1, 16)

        self.layer2_rarm_dec = nn.Linear(h2, 2*h1)
        self.layer2_larm_dec = nn.Linear(h2, 2*h1)
        self.layer2_rleg_dec = nn.Linear(h2, 2*h1)
        self.layer2_lleg_dec = nn.Linear(h2, 2*h1)

        self.layer3_arm_dec = nn.Linear(h3, 2*h2)
        self.layer3_leg_dec = nn.Linear(h3, 2*h2)

        self.layer4_dec = nn.Linear(h4, 2*h3)

    def forward(self, h):
        full_body = self.layer4_dec(h)
        layer3_shape = int(full_body.shape[-1]/2)
        upper_body = self.layer3_arm_dec(full_body[..., :layer3_shape])
        lower_body = self.layer3_leg_dec(full_body[..., -layer3_shape:])

        layer2_shape = int(upper_body.shape[-1]/2)

        right_arm_layer2 = self.layer2_rarm_dec(upper_body[..., :layer2_shape])
        left_arm_layer2 = self.layer2_larm_dec(upper_body[..., -layer2_shape:])
        right_leg_layer2 = self.layer2_rleg_dec(lower_body[..., :layer2_shape])
        left_leg_layer2 = self.layer2_lleg_dec(lower_body[..., -layer2_shape:])

        layer1_shape = int(right_arm_layer2.shape[-1]/2)

        right_arm_layer1 = self.layer1_rarm_dec(
            right_arm_layer2[..., :layer1_shape])
        torso_1_layer1 = self.layer1_torso_dec(
            right_arm_layer2[..., -layer1_shape:])

        left_arm_layer1 = self.layer1_larm_dec(
            left_arm_layer2[..., :layer1_shape])
        torso_2_layer1 = self.layer1_torso_dec(
            left_arm_layer2[..., -layer1_shape:])

        right_leg_layer1 = self.layer1_rleg_dec(
            right_leg_layer2[..., :layer1_shape])
        torso_3_layer1 = self.layer1_torso_dec(
            right_leg_layer2[..., -layer1_shape:])

        left_leg_layer1 = self.layer1_lleg_dec(
            left_leg_layer2[..., :layer1_shape])
        torso_4_layer1 = self.layer1_torso_dec(
            left_leg_layer2[..., -layer1_shape:])
        torso = (torso_1_layer1 + torso_2_layer1 +
                 torso_3_layer1 + torso_4_layer1)/4.
        torso_trunk = torso[..., :13]
        torso_root = torso[..., -3:]
        pred = torch.cat(
            (torso_trunk, left_arm_layer1, right_arm_layer1,
             left_leg_layer1, right_leg_layer1, torso_root), dim=-1)

        return pred


class PoseGenerator(nn.Module):
    def __init__(self, chunks, input_size=300, Seq2SeqKwargs={}, load=None):
        super(PoseGenerator, self).__init__()
        self.chunks = chunks
        self.h1 = 32
        self.h2 = 128
        self.h3 = 512
        self.h4 = 1024

        self.trajectory_size = Seq2SeqKwargs['trajectory_size']
        self.pose_size = Seq2SeqKwargs['pose_size']

        self.pose_enc = PoseEncoder(self.h1, self.h2, self.h3, self.h4)
        self.vel_dec = VelDecoder(self.h1, self.h2, self.h3, self.h4)

        # self.sentence_enc = LSTMEncoder(self.h4)
        self.sentence_enc = BERTSentenceEncoder(self.h4)
        if load:
            self.load_state_dict(torch.load(open(load, 'rb')))
            print('PoseGenerator Model Loaded')
        else:
            print('PoseGenerator Model initialising randomly')

    def train_pose(self, P_in, gt, s2v, train=False, epoch=np.inf):
        time_steps = P_in.shape[-2]
        z_p = self.pose_enc(P_in)
        Q_v = self.vel_dec(z_p, time_steps, gt=P_in)
        z_gen_v = self.pose_enc(Q_v)
        manifold_loss = F.smooth_l1_loss(z_gen_v, z_p, reduction='mean')
        reconstruction_loss = F.smooth_l1_loss(Q_v, P_in, reduction='mean')
        velocity_orig = P_in[:, 1:, :] - P_in[:, :-1, :]
        velocity_Q_v = Q_v[:, 1:, :] - Q_v[:, :-1, :]
        velocity_loss = F.smooth_l1_loss(velocity_orig, velocity_Q_v)
        internal_losses = [0.001*manifold_loss, reconstruction_loss,
                           0.1*velocity_loss]

    def forward(self, P_in, gt, s2v, train=False, epoch=np.inf):
        '''trajectory separate column 0'''
        time_steps = P_in.shape[-2]
        z_p = self.pose_enc(P_in)
        Q_v = self.vel_dec(z_p, time_steps, gt=P_in)
        z_gen_v = self.pose_enc(Q_v)
        manifold_loss = F.smooth_l1_loss(z_gen_v, z_p, reduction='mean')
        reconstruction_loss = F.smooth_l1_loss(Q_v, P_in, reduction='mean')

        language_z, _ = self.sentence_enc(s2v)

        Q_v_lang = self.vel_dec(language_z, time_steps, gt=P_in)

        encoder_loss = F.smooth_l1_loss(
            z_p, language_z, reduction='mean')

        reconstruction_loss_lang = F.smooth_l1_loss(
            Q_v_lang, P_in, reduction='mean')

        # velocity constraint between two frames
        velocity_orig = P_in[:, 1:, :] - P_in[:, :-1, :]
        velocity_Q_v = Q_v[:, 1:, :] - Q_v[:, :-1, :]
        velocity_Q_v_lang = Q_v_lang[:, 1:, :] - Q_v_lang[:, :-1, :]
        velocity_loss = F.smooth_l1_loss(velocity_orig, velocity_Q_v) + \
            F.smooth_l1_loss(velocity_orig, velocity_Q_v_lang)

        internal_losses = [0.001*manifold_loss, 0.1*encoder_loss,
                           reconstruction_loss, reconstruction_loss_lang,
                           0.1*velocity_loss]

        return Q_v_lang, internal_losses

    def sample(self, s2v, time_steps, start):
        ''' last four columns of P_in is always 0 in input so no need to train it. Just ouput zero for columns 63,64,65, 66'''
        start = start[..., :-4]
        language_z, _ = self.sentence_enc(s2v)
        Q_v_lang = self.vel_dec.sample(language_z, time_steps, start)
        tz = torch.zeros((Q_v_lang.shape[0], Q_v_lang.shape[1], 4)).to(
            Q_v_lang.device).double()

        predicted_pose = torch.cat((Q_v_lang, tz), dim=-1)
        # print(predicted_pose.shape)
        return predicted_pose, []

    def sample_encoder(self, s2v, x):
        ''' last four columns of P_in is always 0 in input so no need to train it. Just ouput zero for columns 63,64,65, 66'''
        P_in = x[..., :-4]
        language_z, _ = self.sentence_enc(s2v)
        z = self.pose_enc(P_in)
        gs = language_z * torch.transpose(language_z, 0, 1)
        gp = z * torch.transpose(z, 0, 1)
        # print(predicted_pose.shape)
        # return gp, gs
        return z, language_z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.net = nn.Sequential(nn.Linear(64, 256),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(128, 64),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(64, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        x, h = self.rnn(x)
        out = self.net(h)
        return out.squeeze(0)
