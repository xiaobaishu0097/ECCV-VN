from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput


class MetaMemoryGraphModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(MetaMemoryGraphModel, self).__init__()

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)

        self.graph_detection_feature = nn.Sequential(
            nn.Linear(518, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.graph_detection_other_info_linear_1 = nn.Linear(6, self.num_cate)
        self.graph_detection_other_info_linear_2 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_3 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_4 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_5 = nn.Linear(self.num_cate, self.num_cate)

        self.embed_action = nn.Linear(action_space, 10)

        pointwise_in_channels = 64 + self.num_cate + 10

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.softmax = nn.Softmax(dim=1)

        self.multi_heads = args.multi_heads

        self.meta_current_state_embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
        )
        self.meta_current_action_embedding = nn.Linear(6, 6)
        self.meta_memory_embedding = nn.Sequential(
            nn.Linear(518, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 518),
            nn.LayerNorm(518)
        )

        self.meta_learning_residual_block = nn.Sequential(
            nn.Linear(1030, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1030),
            nn.LayerNorm(1030)
        )
        self.meta_learning_predict = nn.Sequential(
            nn.Linear(1030, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_embedding_input):
        target_info = torch.cat((target['info'], target['indicator']), dim=1)
        target_info = F.relu(self.graph_detection_other_info_linear_1(target_info))
        target_info = target_info.t()
        target_info = F.relu(self.graph_detection_other_info_linear_2(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_3(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_4(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_5(target_info))
        target_appear = torch.mm(target['appear'].t(), target_info).t()
        target = torch.cat((target_appear, target['info'], target['indicator']), dim=1)

        target = F.relu(self.graph_detection_feature(target))
        target_embedding = target.reshape(1, self.num_cate, 7, 7)

        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

        image_embedding = F.relu(self.conv1(state))
        x = self.dropout(image_embedding)

        x = torch.cat((x, target_embedding, action_reshaped), dim=1)
        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c, action_probs, states_rep, states_memory, actions_memory,
                top_k=10):

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])

        current_state_rep = F.relu(torch.add(x, self.meta_current_state_embedding(x)))
        if not torch.eq(action_probs, 0).all():
            last_state_memory = current_state_rep
            last_action_memory = F.relu(self.meta_current_action_embedding(action_probs))
            if not torch.eq(states_memory, 0).all():
                states_memory = torch.cat((states_memory, last_state_memory), dim=0)
                actions_memory = torch.cat((actions_memory, last_action_memory), dim=0)
            else:
                states_memory = last_state_memory
                actions_memory = last_action_memory
        else:
            last_state_memory = None
            last_action_memory = None

        attention_state_memory = current_state_rep
        for step in range(self.multi_heads):
            match_scores = torch.mm(attention_state_memory, states_rep.T)
            if top_k is not None and match_scores.shape[1] > top_k:
                match_scores, indices_topk = torch.topk(match_scores, top_k, dim=1, sorted=False)
                states_memory_topk = torch.squeeze(states_memory[indices_topk, :])
                actions_memory_topk = torch.squeeze(actions_memory[indices_topk, :])
            else:
                states_memory_topk = states_memory
                actions_memory_topk = actions_memory
            match_scores = self.softmax(match_scores)
            attention_state_memory = torch.mm(match_scores, states_memory_topk)
            attention_action_memory = torch.mm(match_scores, actions_memory_topk)
            attention_memory_step = torch.cat((attention_state_memory, attention_action_memory), dim=1)
            if step == 0:
                attention_memory = attention_memory_step
            else:
                attention_memory = attention_memory + attention_memory_step
        attention_memory = F.relu(self.meta_memory_embedding(attention_memory))

        meta_state_rep = torch.cat((current_state_rep, attention_memory), dim=1)
        meta_state_rep_residual = self.meta_learning_residual_block(meta_state_rep)
        meta_state_rep = F.relu(meta_state_rep + meta_state_rep_residual)
        meta_action_pred = self.meta_learning_predict(meta_state_rep)

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx), current_state_rep, last_state_memory, last_action_memory, meta_action_pred

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        states_rep = model_input.states_rep
        states_memory = model_input.states_memory
        action_memory = model_input.action_memory

        x, image_embedding = self.embedding(state, target, action_probs)
        actor_out, critic_out, (hx, cx), state_rep, state_memory, action_memory, meta_action = self.a3clstm(x, hx, cx, action_probs, states_rep, states_memory, action_memory)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
            state_representation=state_rep,
            state_memory=state_memory,
            action_memory=action_memory,
            meta_action=meta_action,
        )
