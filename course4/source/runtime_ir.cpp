#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kuiper_infer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string &bin_path) {
  this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string &param_path) {
  this->param_path_ = param_path;
}

const std::string &RuntimeGraph::param_path() const {
  return this->param_path_;
}

const std::string &RuntimeGraph::bin_path() const { return this->bin_path_; }

void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand *> &inputs,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator *producer = input->producer;
    std::shared_ptr<RuntimeOperand> runtime_operand =
        std::make_shared<RuntimeOperand>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = input->shape;

    switch (input->type) {
    case 1: {
      runtime_operand->type = RuntimeDataType::kTypeFloat32;
      break;
    }
    case 0: {
      runtime_operand->type = RuntimeDataType::kTypeUnknown;
      break;
    }
    default: {
      LOG(FATAL) << "Unknown input operand type: " << input->type;
    }
    }
    runtime_operator->input_operands.insert({producer->name, runtime_operand});
    runtime_operator->input_operands_seq.push_back(runtime_operand);
  }
}

void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand *> &outputs,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *output : outputs) {
    if (!output) {
      continue;
    }
    const auto &consumers = output->consumers;
    for (const auto &c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

void RuntimeGraph::InitGraphParams(
    const std::map<std::string, pnnx::Parameter> &params,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &[name, parameter] : params) {
    const int type = parameter.type;
    switch (type) {
    case int(RuntimeParameterType::kParameterUnknown): {
      RuntimeParameter *runtime_parameter = new RuntimeParameter;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }

    case int(RuntimeParameterType::kParameterBool): {
      RuntimeParameterBool *runtime_parameter = new RuntimeParameterBool;
      runtime_parameter->value = parameter.b;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }

    case int(RuntimeParameterType::kParameterInt): {
      RuntimeParameterInt *runtime_parameter = new RuntimeParameterInt;
      runtime_parameter->value = parameter.i;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }

    case int(RuntimeParameterType::kParameterFloat): {
      RuntimeParameterFloat *runtime_parameter = new RuntimeParameterFloat;
      runtime_parameter->value = parameter.f;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }

    case int(RuntimeParameterType::kParameterString): {
      RuntimeParameterString *runtime_parameter = new RuntimeParameterString;
      runtime_parameter->value = parameter.s;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }

    case int(RuntimeParameterType::kParameterIntArray): {
      RuntimeParameterIntArray *runtime_parameter =
          new RuntimeParameterIntArray;
      runtime_parameter->value = parameter.ai;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }

    case int(RuntimeParameterType::kParameterFloatArray): {
      RuntimeParameterFloatArray *runtime_parameter =
          new RuntimeParameterFloatArray;
      runtime_parameter->value = parameter.af;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }
    case int(RuntimeParameterType::kParameterStringArray): {
      RuntimeParameterStringArray *runtime_parameter =
          new RuntimeParameterStringArray;
      runtime_parameter->value = parameter.as;
      runtime_operator->params.insert({name, runtime_parameter});
      break;
    }
    default: {
      LOG(FATAL) << "Unknown parameter type: " << type;
    }
    }
  }
}

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Can not find the param path or bin path: " << param_path_
               << " " << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator *> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  this->operators_.clear();
  this->operators_maps_.clear();
  for (const pnnx::Operator *op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node";
      continue;
    } else {
      std::shared_ptr<RuntimeOperator> runtime_operator =
          std::make_shared<RuntimeOperator>();
      // 初始化算子的名称
      runtime_operator->name = op->name;
      runtime_operator->type = op->type;

      // 初始化算子中的input
      const std::vector<pnnx::Operand *> &inputs = op->inputs;
      if (!inputs.empty()) {
        InitGraphOperatorsInput(inputs, runtime_operator);
      }

      // 记录输出operand中的名称
      const std::vector<pnnx::Operand *> &outputs = op->outputs;
      if (!outputs.empty()) {
        InitGraphOperatorsOutput(outputs, runtime_operator);
      }

      // 初始化算子中的attribute(权重)
      const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
      if (!attrs.empty()) {
        InitGraphAttrs(attrs, runtime_operator);
      }

      // 初始化算子中的parameter
      const std::map<std::string, pnnx::Parameter> &params = op->params;
      if (!params.empty()) {
        InitGraphParams(params, runtime_operator);
      }
      this->operators_.push_back(runtime_operator);
      this->operators_maps_.insert({runtime_operator->name, runtime_operator});
    }
  }

  graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::Build(const std::string &input_name,
                         const std::string &output_name) {
  if (graph_state_ == GraphState::Complete) {
    LOG(INFO) << "Model has been built already!";
    return;
  }

  // 逐步构造，每一阶段的构造中修改状态，在完成之后，应当检查其位于下一阶段
  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
      << "Graph status error, current state is " << int(graph_state_);
  LOG_IF(FATAL, this->operators_.empty())
      << "Graph operators is empty, may be no init";

  // 构建图关系
  for (const auto &current_op : this->operators_) {
    // 获取当前节点的所有后继节点的names，遍历根据next_op_name从operators_maps_中插入所需要的节点
    const std::vector<std::string> &output_names = current_op->output_names;
    for (const auto &kOutputName : output_names) {
      if (const auto &output_op = this->operators_maps_.find(kOutputName);
          output_op != this->operators_maps_.end()) {
        current_op->output_operators.insert({kOutputName, output_op->second});
      }
    }
  }

  // 初始化节点的输入和输出空间
  RuntimeOperatorUtils::InitOperatorInput(operators_);
  RuntimeOperatorUtils::InitOperatorOutput(graph_->ops, operators_);

  // 构建拓扑顺序
  /*
  topo_operators_.clear();
  for (const auto &[_, op] : operators_maps_) {
    // 根据输入节点构建拓扑排序
    if (op->type == "pnnx.Input" && !op->has_forward) {
       this->ReverseTopo(op);
      // 修改1：Bfs，但是错误
      //this->Bfs(op);
    }
  }

  std::reverse(topo_operators_.begin(), topo_operators_.end());
  */

  //方法2：直接拓扑

  this->Topo();
  //添加全面的拓扑排序检查
  CheckTopoQueueValid();

  graph_state_ = GraphState::Complete;
  input_name_ = input_name;
  output_name_ = output_name;
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }
}

void RuntimeGraph::ReverseTopo(
    const std::shared_ptr<RuntimeOperator> &root_op) {
  CHECK(root_op != nullptr) << "current operator is nullptr";
  root_op->has_forward = true;
  const auto &next_ops = root_op->output_operators;
  for (const auto &[_, op] : next_ops) {
    if (op != nullptr) {
      if (!op->has_forward) {
        this->ReverseTopo(op);
      }
    }
  }
  for (const auto &[_, op] : next_ops) {
    CHECK_EQ(op->has_forward, true);
  }
  this->topo_operators_.push_back(root_op);
}

void RuntimeGraph::Bfs(
  const std::shared_ptr<RuntimeOperator> &root_op) {
    CHECK(root_op != nullptr) << "current operator is nullptr";
    //使用BFS 队列实现
    std::queue<std::shared_ptr<RuntimeOperator>> q;
    q.push(root_op);        // 仅拷贝shared_ptr本身，不会拷贝RuntimeOperator对象
    root_op->has_forward = true;
    while(!q.empty()){
      auto cur = q.front();
      q.pop();
      this->topo_operators_.push_back(cur);
      for(auto &[_,op]:cur->output_operators){
        if(op!=nullptr && !op->has_forward){
          op->has_forward = true;
          q.push(op);
        }
      }
    }
    for(auto &op:this->operators_){       //检查是否均加入
      CHECK_EQ(op->has_forward,true);
    }

}

void RuntimeGraph::Topo() {
  // 重新构建拓扑前，先清理遍历标记。
  for (const auto &op : operators_) {
    CHECK(op != nullptr) << "Meet nullptr in operators list";
    op->has_forward = false;
  }

  // 统计每个节点的入度：前驱节点数量。
  std::unordered_map<std::string, size_t> indegree_map;
  indegree_map.reserve(operators_.size());
  for (const auto &op : operators_) {
    indegree_map[op->name] = 0;
  }

  for (const auto &op : operators_) {
    for (const auto &[next_name, next_op] : op->output_operators) {
      CHECK(next_op != nullptr)
          << "Meet nullptr output operator from: " << op->name;
      CHECK(indegree_map.count(next_name))
          << "Output operator missing in operators map: " << next_name;
      indegree_map[next_name] += 1;
    }
  }

  // Kahn算法：先把所有入度为0的节点入队。
  std::queue<std::shared_ptr<RuntimeOperator>> zero_indegree_queue;
  for (const auto &op : operators_) {
    CHECK(indegree_map.count(op->name))
        << "Operator missing in indegree map: " << op->name;
    if (indegree_map.at(op->name) == 0) {
      zero_indegree_queue.push(op);
    }
  }

  while (!zero_indegree_queue.empty()) {
    std::shared_ptr<RuntimeOperator> cur_op = zero_indegree_queue.front();
    zero_indegree_queue.pop();

    topo_operators_.push_back(cur_op);
    cur_op->has_forward = true;

    for (const auto &[next_name, next_op] : cur_op->output_operators) {
      CHECK(next_op != nullptr)
          << "Meet nullptr output operator from: " << cur_op->name;
      CHECK(indegree_map.count(next_name))
          << "Output operator missing in indegree map: " << next_name;
      CHECK_GT(indegree_map.at(next_name), 0)
          << "Invalid indegree for operator: " << next_name;

      indegree_map[next_name] -= 1;
      if (indegree_map.at(next_name) == 0) {
        zero_indegree_queue.push(next_op);
      }
    }
  }

  CHECK_EQ(topo_operators_.size(), operators_.size())
      << "Topo sort failed, graph may contain cycle or disconnected relation";
}

void RuntimeGraph::CheckTopoQueueValid() const {
  CHECK_EQ(topo_operators_.size(), operators_.size())     //空间一致
      << "Topo queue size mismatch with operators";

  std::unordered_map<std::string, size_t> index_map;
  index_map.reserve(topo_operators_.size());        //预留空间，加入所有topu_op
  for (size_t i = 0; i < topo_operators_.size(); ++i) {     //加入哈希，并检查不空不重
    const auto &op = topo_operators_.at(i);
    CHECK(op != nullptr) << "Meet nullptr in topo queue";
    const auto [_, inserted] = index_map.insert({op->name, i});
    CHECK(inserted) << "Duplicate operator in topo queue: " << op->name;
  }

  for (const auto &op : operators_) {
    CHECK(op != nullptr) << "Meet nullptr in operators list";
    CHECK(index_map.count(op->name))
        << "Operator missing in topo queue: " << op->name;    //op存在
    for (const auto &[next_name, next_op] : op->output_operators) {
      CHECK(next_op != nullptr)
          << "Meet nullptr output operator from: " << op->name;
      CHECK(index_map.count(next_name))
          << "Output operator missing in topo queue: " << next_name;
      CHECK_LT(index_map.at(op->name), index_map.at(next_name))     //核心：前驱索引小于等于后继
          << "Invalid topo relation: " << op->name << " should be before "
          << next_name;
    }
  }
}

void RuntimeGraph::InitGraphAttrs(
    const std::map<std::string, pnnx::Attribute> &attrs,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &[name, attr] : attrs) {
    switch (attr.type) {
    case 1: {
      std::shared_ptr<RuntimeAttribute> runtime_attribute =
          std::make_shared<RuntimeAttribute>();
      runtime_attribute->type = RuntimeDataType::kTypeFloat32;
      runtime_attribute->weight_data = attr.data;
      runtime_attribute->shape = attr.shape;
      runtime_operator->attribute.insert({name, runtime_attribute});
      break;
    }
    default: {
      LOG(FATAL) << "Unknown attribute type: " << attr.type;
    }
    }
  }
}

const std::vector<std::shared_ptr<RuntimeOperator>> &
RuntimeGraph::operators() const {
  return this->operators_;
}

const std::vector<std::shared_ptr<RuntimeOperator>> &
RuntimeGraph::get_topo_queues() const {
  return this->topo_operators_;
}

RuntimeGraph::GraphState RuntimeGraph::graph_state() const { return this->graph_state_; }

} // namespace kuiper_infer
