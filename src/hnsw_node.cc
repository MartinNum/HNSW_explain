// Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "n2/hnsw_node.h"

namespace n2 {

HnswNode::HnswNode(int id, const Data* data, int level, int maxsize, int maxsize0)
: id_(id), data_(data), level_(level), maxsize_(maxsize), maxsize0_(maxsize0) {
    // 初始化节点层数
    friends_at_layer_.resize(level+1);
    // 初始化每层的邻居数（除0层）
    for (int i = 1; i <= level; ++i) {
        friends_at_layer_[i].reserve(maxsize_ + 1);
    }
    // 初始化0层的邻居数
    friends_at_layer_[0].reserve(maxsize0_ + 1);
}

// 预先分配0层以上每层的内存
void HnswNode::CopyHigherLevelLinksToOptIndex(char* mem_offset, long long memory_per_node_higher_level) const {
    // 1层的起始地址
    char* mem_data = mem_offset;
    for (int level = 1; level <= level_; ++level) {
        // 分配单层的内存
        CopyLinksToOptIndex(mem_data, level);
        // 起始地址加上本层的内存大小（(friends_at_layer_[level].size()+1)*sizeof(int)）
        mem_data += memory_per_node_higher_level;
    }
}
// 预先分配0层的内存
void HnswNode::CopyDataAndLevel0LinksToOptIndex(char* mem_offset, int higher_level_offset, int M0) const {
    // 0层的起始地址
    char* mem_data = mem_offset;
    *((int*)(mem_data)) = higher_level_offset;
    mem_data += sizeof(int);
    // 分配单层的内存
    CopyLinksToOptIndex(mem_data, 0);
    mem_data += (sizeof(int) + sizeof(int)*M0);
    auto& data = data_->GetData();
    // 分陪存储该节点向量的内存
    for (size_t i = 0; i < data.size(); ++i) {
        *((float*)(mem_data)) = (float)data[i];
        mem_data += sizeof(float);
    }
}

void HnswNode::CopyLinksToOptIndex(char* mem_offset, int level) const {
    // 本层存储开始的起始地址
    char* mem_data = mem_offset;
    const auto& neighbors = friends_at_layer_[level];
    // mem_data指向的地址存储内容为邻居的数量
    *((int*)(mem_data)) = (int)(neighbors.size());
    // 本层的起始地址加上邻居的数量所需要的内存空间（sizeof(int)）即为下一步的起始地址
    mem_data += sizeof(int);
    // 从地址mem_data开始将每个邻居的id（int）一次存入
    for (size_t i = 0; i < neighbors.size(); ++i) {
        *((int*)(mem_data)) = (int)neighbors[i]->GetId();
        mem_data += sizeof(int);
    }
}

} // namespace n2
