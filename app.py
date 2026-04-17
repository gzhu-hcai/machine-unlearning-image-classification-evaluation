import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression

# 导入我们的 mock utils
from munl.utils import DictConfig
from munl.settings import DEFAULT_DEVICE

# 动态导入你的 6 个算法
from algorithms.finetune import FinetuneUnlearner
from algorithms.fisher import FisherForgetting
from algorithms.gradient_ascent import GradientAscent
from algorithms.negative_gradient import NegativeGradient
from munl.unlearning.salun import SaliencyUnlearning
from algorithms.successive_random_labels import SuccessiveRandomLabels

# ---------------------------------------------------------
# 1. UI 初始化、基础功能定义与全局种子固化
# ---------------------------------------------------------
st.set_page_config(page_title="机器遗忘评测系统", layout="wide")
st.title("机器遗忘算法图形化评测系统")

# === 初始化历史记录记忆体 ===
if 'history' not in st.session_state:
    st.session_state.history = []

# === 数据集与评测指标详细说明 ===
with st.expander("数据集与评测指标详细说明 (点击展开)", expanded=False):
    st.markdown("""
    ###数据集说明
    * **来源**: CIFAR-10 / CIFAR-100 官方数据集
    * **类型**: 10 或 100 分类彩色图像 (32x32 分辨率)
    * **规模**: 包含 50,000 张训练集图片和 10,000 张测试集图片
    * **预处理**: 图像已进行 Normalize 标准化处理。系统会根据左侧设定的 `Forget Ratio` 动态将原训练集划分为**保留集 (Retain Set)**和**遗忘集 (Forget Set)**。

    ###测试指标说明
    * **Retain Acc (保留集准确率)**: 评估算法对正常数据的“通用效用保护”能力，**该值越高、越接近原模型越好**。
    * **Forget Acc (遗忘集准确率)**: 评估算法的“遗忘有效性”，**该值越低越好**（说明模型成功剥离了记忆）。
    * **MIA 成功率**: 评估模型的隐私防御能力（成员推理攻击），**该值越接近 50% 越好**（等效于 AUC 值，说明攻击者只能盲猜）。
    * **L2 参数距离**: 评估遗忘前后模型权重的变化幅度，**该值越小越好**（说明只做了微创手术）。
    """)


# === 全局随机种子固化函数 ===
def set_seed(seed=42):
    """固定所有的随机种子以保证实验的绝对可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@st.cache_resource
def get_dataloaders(dataset_name, batch_size=128, forget_ratio=0.1, seed=42):
    """根据选择动态加载数据集，并接受 Seed 控制划分"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset_name == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    total_train = len(train_dataset)
    forget_size = int(total_train * forget_ratio)
    retain_size = total_train - forget_size

    generator = torch.Generator().manual_seed(seed)
    retain_dataset, random_split_forget = random_split(train_dataset, [retain_size, forget_size], generator=generator)

    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    forget_loader = DataLoader(random_split_forget, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return retain_loader, forget_loader, val_loader


def load_model(arch_name, num_classes=10, weights_path=None):
    if arch_name == "ResNet-18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        expected_layer_name = 'fc.weight'

    # === 【更新】：TinyViT 强行组装逻辑 ===
    elif arch_name == "TinyViT-11M":
        try:
            # 我们不找那个不存在的名字了，直接导入核心构造器
            from tiny_vit import _create_tiny_vit
        except Exception as e:
            st.error("❌ 导入模型文件失败！")
            st.error(f"真正的底层报错原因是: {e}")
            st.stop()

        # 手动写死 11M 版本的专属架构参数，并根据 CIFAR 修改 img_size=32
        model_kwargs = dict(
            embed_dims=[64, 128, 256, 448],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 8, 14],
            window_sizes=[7, 7, 14, 7],
            num_classes=num_classes,
            img_size=32
        )
        # 强行构建模型
        model = _create_tiny_vit('tiny_vit_11m_224', pretrained=False, **model_kwargs)
        expected_layer_name = 'head.weight'

    else:
        raise ValueError(f"不受支持的模型架构: {arch_name}")

    if weights_path is not None:
        try:
            state_dict = torch.load(weights_path, map_location=DEFAULT_DEVICE)

            # 剥离嵌套字典
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'net' in state_dict:
                state_dict = state_dict['net']

            # 自动修复多显卡训练带来的 'module.' 前缀问题
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v
            state_dict = clean_state_dict

            # 哨兵逻辑：严格检查分类数
            if expected_layer_name in state_dict:
                ckpt_classes = state_dict[expected_layer_name].shape[0]
                if ckpt_classes != num_classes:
                    raise ValueError(f"❌ 分类数发生冲突！您当前选择的数据集需要 {num_classes} 分类，"
                                     f"但您上传的预训练权重是 {ckpt_classes} 分类的。")

            model.load_state_dict(state_dict)
            st.toast(f"✅ 成功加载纯正的 {num_classes} 分类 {arch_name} 预训练模型！", icon="🎉")

        except Exception as e:
            st.error(f"加载模型权重失败: {e}")
            st.error(f"调试提示: 您上传的文件里实际包含的键名大概长这样 -> {list(state_dict.keys())[:5]}")
            st.stop()

    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def calculate_l2_distance(model_orig, model_new):
    distance = 0.0
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(model_orig.named_parameters(), model_new.named_parameters()):
            if param1.requires_grad and param2.requires_grad:
                distance += torch.sum((param1 - param2) ** 2).item()
    return distance ** 0.5


def evaluate_mia(model, retain_loader, forget_loader, test_loader, device):
    model.eval()

    def get_entropy(loader):
        entropies = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                outputs = torch.nn.functional.softmax(model(inputs), dim=1)
                entropy = -torch.sum(outputs * torch.log(outputs + 1e-8), dim=1)
                entropies.extend(entropy.cpu().numpy())

        clean_entropies = np.nan_to_num(np.array(entropies), nan=0.0, posinf=0.0, neginf=0.0)
        return clean_entropies.reshape(-1, 1)

    retain_features = get_entropy(retain_loader)
    test_features = get_entropy(test_loader)
    forget_features = get_entropy(forget_loader)

    min_len = min(len(retain_features), len(test_features))
    X_train = np.vstack((retain_features[:min_len], test_features[:min_len]))
    y_train = np.hstack((np.ones(min_len), np.zeros(min_len)))

    attacker = LogisticRegression(random_state=42)
    attacker.fit(X_train, y_train)

    preds = attacker.predict(forget_features)
    mia_success_rate = np.mean(preds == 1) * 100.0
    return mia_success_rate


# ---------------------------------------------------------
# 2. 左侧边栏：参数配置面板
# ---------------------------------------------------------
st.sidebar.header("参数配置")
dataset_name = st.sidebar.selectbox("数据集", ["CIFAR-10", "CIFAR-100"])

st.sidebar.subheader("模型加载")
# === 【更新】：去掉了 VGG-16，加入了 TinyViT-11M ===
model_name = st.sidebar.selectbox("模型架构", ["ResNet-18", "TinyViT-11M"])
uploaded_model_file = st.sidebar.file_uploader("上传纯正的预训练模型 (.pth)", type=["pth", "pt"])

algo_map = {
    "Finetune Unlearner": FinetuneUnlearner,
    "Fisher Forgetting": FisherForgetting,
    "Gradient Ascent": GradientAscent,
    "Negative Gradient": NegativeGradient,
    "Saliency Unlearning (SalUn)": SaliencyUnlearning,
    "Successive Random Labels": SuccessiveRandomLabels
}
algo_choice = st.sidebar.selectbox("遗忘算法", list(algo_map.keys()))

st.sidebar.subheader("通用超参数")
random_seed = st.sidebar.number_input("随机种子 (Seed)", min_value=0, max_value=999999, value=42, step=1,
                                      help="固定种子以确保每次实验数据划分与模型初始化绝对一致")
epochs = st.sidebar.slider("Epochs (迭代次数)", min_value=1, max_value=50, value=2)
lr = st.sidebar.number_input("Learning Rate (学习率)", value=0.01, format="%.4f")
batch_size = st.sidebar.selectbox("Batch Size", [8, 32, 64, 128], index=1)
forget_ratio = st.sidebar.slider("Forget Ratio (遗忘数据占比)", 0.01, 0.50, 0.10)

extra_cfg = {}
st.sidebar.subheader("🔧 算法专属参数")
if algo_choice == "Saliency Unlearning (SalUn)":
    extra_cfg["threshold"] = st.sidebar.slider("Threshold (显著性阈值)", 0.1, 1.0, 0.5)
elif algo_choice == "Fisher Forgetting":
    extra_cfg["alpha"] = st.sidebar.slider("Alpha (噪声系数)", 0.0, 1.0, 0.2)
else:
    st.sidebar.caption("当前算法无额外专属参数。")

# ---------------------------------------------------------
# 3. 主界面：实验控制与执行逻辑
# ---------------------------------------------------------
if st.button("一键执行遗忘操作 (Run Evaluation)", use_container_width=True):

    if uploaded_model_file is None:
        st.warning("请先在左侧上传对应的预训练模型权重！")
        st.stop()

    set_seed(random_seed)

    weights_path = os.path.join(".", uploaded_model_file.name)
    with open(weights_path, "wb") as f:
        f.write(uploaded_model_file.getbuffer())

    num_classes = 10 if dataset_name == "CIFAR-10" else 100

    with st.spinner(f"正在严格校验权重并加载 {num_classes} 分类 {model_name}..."):
        model = load_model(arch_name=model_name, num_classes=num_classes, weights_path=weights_path).to(DEFAULT_DEVICE)

    with st.spinner(f"正在加载并划分真实的 {dataset_name} 数据集... (Seed: {random_seed})"):
        retain_loader, forget_loader, val_loader = get_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            forget_ratio=forget_ratio,
            seed=random_seed
        )

    cfg = DictConfig({
        "num_epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "seed": random_seed,
        **extra_cfg
    })

    UnlearnerClass = algo_map[algo_choice]
    unlearner = UnlearnerClass(cfg=cfg, device=DEFAULT_DEVICE)

    with st.spinner("正在评估原始模型性能及 MIA 基准..."):
        orig_retain_acc = evaluate_model(model, retain_loader, DEFAULT_DEVICE)
        orig_forget_acc = evaluate_model(model, forget_loader, DEFAULT_DEVICE)
        orig_test_acc = evaluate_model(model, val_loader, DEFAULT_DEVICE)
        orig_mia = evaluate_mia(model, retain_loader, forget_loader, val_loader, DEFAULT_DEVICE)

    with st.spinner(f"正在使用 {algo_choice} 算法执行机器遗忘 (这可能需要一些时间)..."):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        unlearned_model = unlearner.unlearn(model, retain_loader, forget_loader, val_loader)

        end_time = time.time()
        time_cost = end_time - start_time

        if torch.cuda.is_available():
            max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            max_memory_mb = 0.0

    st.success(f"✅ {algo_choice} 执行完毕！计算耗时: {time_cost:.2f} 秒 | 显存峰值消耗: {max_memory_mb:.2f} MB")

    with st.spinner("正在评估遗忘后模型的真实性能及新 MIA 成功率..."):
        new_retain_acc = evaluate_model(unlearned_model, retain_loader, DEFAULT_DEVICE)
        new_forget_acc = evaluate_model(unlearned_model, forget_loader, DEFAULT_DEVICE)
        new_test_acc = evaluate_model(unlearned_model, val_loader, DEFAULT_DEVICE)
        new_mia = evaluate_mia(unlearned_model, retain_loader, forget_loader, val_loader, DEFAULT_DEVICE)

    with st.spinner("正在计算模型参数 L2 距离..."):
        orig_clean_model = load_model(arch_name=model_name, num_classes=num_classes, weights_path=weights_path).to(
            DEFAULT_DEVICE)
        l2_dist = calculate_l2_distance(orig_clean_model, unlearned_model)

    delta_retain = new_retain_acc - orig_retain_acc
    delta_forget = new_forget_acc - orig_forget_acc
    delta_test = new_test_acc - orig_test_acc
    delta_mia = new_mia - orig_mia

    # ---------------------------------------------------------
    # 4. 实验报告与数据可视化
    # ---------------------------------------------------------
    st.markdown("---")
    st.header("当前实验报告与分析")

    # ================= 新增：智能裁判引擎 =================
    st.subheader("系统智能判定结果")

    # 判定逻辑阈值设定
    MIA_TARGET_LOW = 45.0
    MIA_TARGET_HIGH = 55.0
    RETAIN_DROP_TOLERANCE = -5.0  # 保留集最多允许掉 5%

    status_box = st.empty()

    if delta_retain < -10.0 or new_retain_acc < 50.0:
        # 1. 灾难性崩溃 (模型被破坏)
        status_box.error(
            f"**❌ 判定结果：灾难性遗忘 (Catastrophic Forgetting)**\n\n"
            f"**诊断分析**：保留集准确率大幅下降了 {abs(delta_retain):.2f}%！算法过度破坏了底层特征空间，"
            f"虽然可能达成了遗忘，但模型已经丧失了基础任务能力，该方案在工业界**不可用**。"
        )
    elif new_mia > MIA_TARGET_HIGH:
        # 2. 遗忘不彻底 (隐私泄露风险大)
        status_box.warning(
            f"**⚠️ 判定结果：遗忘不彻底 (Incomplete Unlearning)**\n\n"
            f"**诊断分析**：MIA 攻击成功率高达 {new_mia:.2f}%！尽管保留集效用完好，"
            f"但深层网络依然残留着遗忘集的‘暗知识’，容易被黑客提取出隐私数据。"
        )
    elif new_mia < MIA_TARGET_LOW:
        # 3. 过度遗忘 (产生统计学黑洞)
        status_box.warning(
            f"**⚠️ 判定结果：过度遗忘 (Over-Unlearning)**\n\n"
            f"**诊断分析**：MIA 攻击成功率低至 {new_mia:.2f}%！这是一种反向隐私泄露。算法在此处留下了"
            f"明显的‘清洗痕迹’（预测置信度异常低），攻击者一扫就能断定该数据曾被特殊处理过。"
        )
    else:
        # 4. 帕累托最优 (完美状态)
        status_box.success(
            f"**✅ 判定结果：完美遗忘 (Ideal Unlearning)**\n\n"
            f"**诊断分析**：太棒了！MIA 成功率被精准压制在盲猜区间（{new_mia:.2f}%），"
            f"同时保留集效用仅微弱波动（{delta_retain:.2f}%）。模型在达成物理级隐私剥离的同时，"
            f"完美维持了正常业务能力！"
        )
    # ===================================================

    st.subheader("1. 核心指标概览")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Retain Acc", f"{new_retain_acc:.2f}%", f"{delta_retain:.2f}%")
    kpi2.metric("Forget Acc", f"{new_forget_acc:.2f}%", f"{delta_forget:.2f}%", delta_color="inverse")
    kpi3.metric("Test Acc", f"{new_test_acc:.2f}%", f"{delta_test:.2f}%")

    # 针对 MIA 指标变色提醒
    mia_color = "normal" if MIA_TARGET_LOW <= new_mia <= MIA_TARGET_HIGH else "inverse"
    kpi4.metric("MIA 成功率", f"{new_mia:.2f}%", f"{delta_mia:.2f}%", delta_color=mia_color,
                help="45%~55%为最佳盲猜区间")

    kpi5.metric("L2 参数距离", f"{l2_dist:.4f}", "越小越好", delta_color="off")
    kpi6.metric("算法耗时", f"{time_cost:.1f}s", "秒")

    st.subheader("2. 本次结果可视化")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        acc_df = pd.DataFrame({
            "指标": ["Retain Acc", "Forget Acc", "Test Acc", "MIA Success Rate"],
            "Original Model": [orig_retain_acc, orig_forget_acc, orig_test_acc, orig_mia],
            "Unlearned Model": [new_retain_acc, new_forget_acc, new_test_acc, new_mia]
        })
        acc_melt = acc_df.melt(id_vars="指标", var_name="Model Status", value_name="Percentage (%)")
        fig_acc = px.bar(acc_melt, x="指标", y="Percentage (%)", color="Model Status", barmode="group",
                         text_auto='.1f', title="遗忘前后多维度真实对比",
                         color_discrete_sequence=["#636EFA", "#EF553B"])
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_chart2:
        drop_df = pd.DataFrame({
            "Metric": ["Retain Drop", "Forget Drop", "Test Drop", "MIA Drop"],
            "Change (%)": [abs(delta_retain), abs(delta_forget), abs(delta_test), abs(delta_mia)]
        })
        fig_drop = px.bar(drop_df, x="Metric", y="Change (%)", text_auto='.1f', title="变化幅度分析",
                          color="Metric", color_discrete_sequence=["#00CC96", "#AB63FA", "#FFA15A", "#FF6692"])
        st.plotly_chart(fig_drop, use_container_width=True)

    report_data = {
        "实验时间": datetime.datetime.now().strftime("%H:%M:%S"),
        "数据集": dataset_name,
        "模型架构": model_name,
        "遗忘算法": algo_choice,
        "Seed": random_seed,
        "Forget Ratio": f"{forget_ratio * 100}%",
        "Epochs": epochs,
        "Learning Rate": lr,
        "Batch Size": batch_size,
        "原 Retain Acc": round(orig_retain_acc, 2),
        "原 Forget Acc": round(orig_forget_acc, 2),
        "原 MIA 成功率": round(orig_mia, 2),
        "新 Retain Acc": round(new_retain_acc, 2),
        "新 Forget Acc": round(new_forget_acc, 2),
        "新 MIA 成功率": round(new_mia, 2),
        "L2 参数距离": round(l2_dist, 4),
        "耗时(秒)": round(time_cost, 2),
        "显存消耗(MB)": round(max_memory_mb, 2)
    }

    st.session_state.history.append(report_data)

    st.subheader("3. 综合实验记录导出")
    df_report = pd.DataFrame([report_data])
    st.dataframe(df_report, use_container_width=True)
    csv = df_report.to_csv(index=False).encode('utf-8-sig')
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(label="一键下载完整实验报告 (CSV)", data=csv,
                       file_name=f"Unlearning_Report_{algo_choice}_{timestamp_str}.csv", mime="text/csv",
                       use_container_width=True)

# ---------------------------------------------------------
# 5. 高级图表：横向多算法对比大屏
# ---------------------------------------------------------
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.header("多次实验结果横向对比大屏 (高级可视化)")

    history_df = pd.DataFrame(st.session_state.history)
    history_df["实验Run"] = history_df["遗忘算法"] + " (Ep:" + history_df["Epochs"].astype(str) + " Sd:" + history_df[
        "Seed"].astype(str) + ")"

    col_adv1, col_adv2 = st.columns(2)

    with col_adv1:
        fig_mia_curve = go.Figure()
        fig_mia_curve.add_trace(go.Scatter(x=history_df["实验Run"], y=history_df["原 MIA 成功率"],
                                           mode='lines+markers', name='原始 MIA 成功率 (遗忘前)',
                                           line=dict(dash='dash', color='gray')))
        fig_mia_curve.add_trace(go.Scatter(x=history_df["实验Run"], y=history_df["新 MIA 成功率"],
                                           mode='lines+markers', name='新 MIA 成功率 (遗忘后)',
                                           line=dict(color='#FF4B4B', width=3), marker=dict(size=10)))
        fig_mia_curve.update_layout(title="MIA 成员推理攻击结果演变曲线", xaxis_title="运行批次",
                                    yaxis_title="MIA 成功率 (%)", hovermode="x unified")
        st.plotly_chart(fig_mia_curve, use_container_width=True)

    with col_adv2:
        fig_eff = make_subplots(specs=[[{"secondary_y": True}]])
        fig_eff.add_trace(
            go.Bar(x=history_df["实验Run"], y=history_df["耗时(秒)"], name="计算耗时 (秒)", marker_color="#636EFA",
                   opacity=0.8), secondary_y=False)
        fig_eff.add_trace(go.Scatter(x=history_df["实验Run"], y=history_df["显存消耗(MB)"], name="显存消耗 (MB)",
                                     mode="lines+markers", marker_color="#00CC96", line=dict(width=3),
                                     marker=dict(size=10)), secondary_y=True)
        fig_eff.update_layout(title="算法计算效率与资源消耗对比图")
        fig_eff.update_yaxes(title_text="耗时 (秒) ⬇越小越好", secondary_y=False)
        fig_eff.update_yaxes(title_text="显存消耗 (MB) ⬇越小越好", secondary_y=True)
        st.plotly_chart(fig_eff, use_container_width=True)

    if st.button("清空历史对比数据"):
        st.session_state.history = []
        st.rerun()
