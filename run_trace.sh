#!/bin/bash

# SparseDrive Trace快速启动脚本
# 使用方法: ./run_trace.sh [脚本名称] [额外参数]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查nsys
    if ! command -v nsys &> /dev/null; then
        print_error "nsys命令未找到，请先安装NVIDIA Nsight Systems"
        print_info "安装命令: sudo dpkg -i NsightSystems-linux-cli-public-*.deb"
        exit 1
    fi
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "Python命令未找到"
        exit 1
    fi
    
    # 检查trace配置文件
    if [ ! -f "sparsedrive_trace.json" ]; then
        print_error "Trace配置文件 sparsedrive_trace.json 不存在"
        exit 1
    fi
    
    print_success "依赖检查通过"
}

# 显示帮助信息
show_help() {
    echo "SparseDrive Trace快速启动脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [脚本名称] [额外参数]"
    echo ""
    echo "参数:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -t, --test     运行测试脚本"
    echo "  -c, --config   显示trace配置"
    echo "  -v, --view     查看最近的trace报告"
    echo "  -s1, --stage1  运行Stage1训练trace"
    echo "  -s2, --stage2  运行Stage2训练trace"
    echo ""
    echo "示例:"
    echo "  $0 -t                    # 运行测试脚本"
    echo "  $0 your_script.py        # 运行自定义脚本"
    echo "  $0 -v                    # 查看trace报告"
    echo "  $0 -s1                   # 运行Stage1训练trace"
    echo "  $0 -s2                   # 运行Stage2训练trace"
    echo ""
}

# 显示trace配置
show_config() {
    print_info "显示trace配置..."
    if [ -f "sparsedrive_trace.json" ]; then
        echo "Trace配置文件内容:"
        cat sparsedrive_trace.json | python -m json.tool
    else
        print_error "Trace配置文件不存在"
    fi
}

# 运行测试脚本
run_test() {
    print_info "运行测试脚本..."
    
    if [ ! -f "test_trace.py" ]; then
        print_error "测试脚本 test_trace.py 不存在"
        exit 1
    fi
    
    print_info "运行基本测试..."
    python test_trace.py
    
    if [ $? -eq 0 ]; then
        print_success "测试完成"
    else
        print_error "测试失败"
        exit 1
    fi
}

# 运行trace分析
run_trace() {
    local script_name=$1
    shift
    
    if [ ! -f "$script_name" ]; then
        print_error "脚本文件 $script_name 不存在"
        exit 1
    fi
    
    print_info "开始trace分析..."
    print_info "脚本: $script_name"
    print_info "参数: $@"
    
    # 生成唯一的输出文件名
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_name="sparsedrive_trace_${timestamp}"
    
    print_info "输出文件: ${output_name}.nsys-rep"
    
    # 运行trace
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --python-functions-trace=sparsedrive_trace.json \
        --output=${output_name} \
        --force-overwrite \
        python "$script_name" "$@"
    
    if [ $? -eq 0 ]; then
        print_success "Trace完成"
        print_info "报告文件: ${output_name}.nsys-rep"
        print_info "使用以下命令查看报告:"
        echo "  nsys-ui ${output_name}.nsys-rep"
        echo "  # 或者"
        echo "  nsys export --type sqlite --output ${output_name}.sqlite ${output_name}.nsys-rep"
    else
        print_error "Trace失败"
        exit 1
    fi
}

# 运行Stage1训练trace
run_stage1() {
    print_info "运行Stage1训练trace..."
    
    local config_file="projects/configs/sparsedrive_small_stage1.py"
    if [ ! -f "$config_file" ]; then
        print_error "Stage1配置文件不存在: $config_file"
        exit 1
    fi
    
    print_info "Stage1配置: $config_file"
    print_info "任务: 检测 + 建图 (无运动规划)"
    
    # 生成唯一的输出文件名
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_name="sparsedrive_stage1_trace_${timestamp}"
    
    print_info "输出文件: ${output_name}.nsys-rep"
    
    # 运行Stage1 trace
    print_info "创建临时配置文件并运行trace..."
    
    # 创建临时配置文件，使用Python脚本安全修改
    local temp_config="/tmp/stage1_trace_config_${timestamp}.py"
    local python_script="/tmp/modify_config_${timestamp}.py"
    
    # 创建Python脚本来修改配置
    cat > "$python_script" << 'EOF'
import re
import sys

def modify_config(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # 使用正则表达式安全替换workers_per_gpu
    # 匹配: workers_per_gpu=batch_size,
    # 替换为: workers_per_gpu=0,  # Set to 0 for profiling
    pattern = r'workers_per_gpu\s*=\s*batch_size\s*,'
    replacement = 'workers_per_gpu=0,  # Set to 0 for profiling'
    
    modified_content = re.sub(pattern, replacement, content)
    
    with open(output_file, 'w') as f:
        f.write(modified_content)
    
    print(f"Configuration modified: {input_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_config output_config")
        sys.exit(1)
    
    modify_config(sys.argv[1], sys.argv[2])
EOF
    
    # 使用Python脚本修改配置
    python3 "$python_script" "$config_file" "$temp_config"
    
    print_info "启动nsys profile..."
    
    nsys profile \
        --python-functions-trace=sparsedrive_trace.json \
        --python-backtrace=cuda \
        --output=${output_name} \
        --trace=cuda,nvtx \
        --delay=15 \
        --duration=30 \
        --force-overwrite=true \
        bash -lc 'cd /workspace/SparseDrive && conda run -n sparsedrive bash -lc "PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh '"$temp_config"' 1 --deterministic"'
    
    # 清理临时文件
    rm -f "$temp_config" "$python_script"
    
    if [ $? -eq 0 ]; then
        print_success "Stage1 Trace完成"
        print_info "报告文件: ${output_name}.nsys-rep"
        print_info "使用以下命令查看报告:"
        echo "  nsys-ui ${output_name}.nsys-rep"
    else
        print_error "Stage1 Trace失败"
        exit 1
    fi
}

# 运行Stage2训练trace
run_stage2() {
    print_info "运行Stage2训练trace..."
    
    local config_file="projects/configs/sparsedrive_small_stage2.py"
    if [ ! -f "$config_file" ]; then
        print_error "Stage2配置文件不存在: $config_file"
        exit 1
    fi
    
    print_info "Stage2配置: $config_file"
    print_info "任务: 检测 + 建图 + 运动规划"
    
    # 检查是否需要先运行Stage1
    print_warning "注意: Stage2需要Stage1的预训练权重"
    print_info "建议先完成Stage1训练，然后加载权重进行Stage2训练"
    
    # 生成唯一的输出文件名
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_name="sparsedrive_stage2_trace_${timestamp}"
    
    print_info "输出文件: ${output_name}.nsys-rep"
    
    # 运行Stage2 trace
    print_info "创建临时配置文件并运行trace..."
    
    # 创建临时配置文件，使用Python脚本安全修改
    local temp_config="/tmp/stage2_trace_config_${timestamp}.py"
    local python_script="/tmp/modify_config_stage2_${timestamp}.py"
    
    # 创建Python脚本来修改配置
    cat > "$python_script" << 'EOF'
import re
import sys

def modify_config(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # 使用正则表达式安全替换workers_per_gpu
    # 匹配: workers_per_gpu=batch_size,
    # 替换为: workers_per_gpu=0,  # Set to 0 for profiling
    pattern = r'workers_per_gpu\s*=\s*batch_size\s*,'
    replacement = 'workers_per_gpu=0,  # Set to 0 for profiling'
    
    modified_content = re.sub(pattern, replacement, content)
    
    with open(output_file, 'w') as f:
        f.write(modified_content)
    
    print(f"Configuration modified: {input_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_config output_config")
        sys.exit(1)
    
    modify_config(sys.argv[1], sys.argv[2])
EOF
    
    # 使用Python脚本修改配置
    python3 "$python_script" "$config_file" "$temp_config"
    
    print_info "启动nsys profile..."
    
    nsys profile \
        --python-functions-trace=sparsedrive_trace.json \
        --python-backtrace=cuda \
        --output=${output_name} \
        --trace=cuda,nvtx \
        --delay=15 \
        --duration=30 \
        --force-overwrite=true \
        bash -lc 'cd /workspace/SparseDrive && conda run -n sparsedrive bash -lc "PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh '"$temp_config"' 1 --deterministic"'
    
    # 清理临时文件
    rm -f "$temp_config" "$python_script"
    
    if [ $? -eq 0 ]; then
        print_success "Stage2 Trace完成"
        print_info "报告文件: ${output_name}.nsys-rep"
        print_info "使用以下命令查看报告:"
        echo "  nsys-ui ${output_name}.nsys-rep"
    else
        print_error "Stage2 Trace失败"
        exit 1
    fi
}

# 查看最近的trace报告
view_reports() {
    print_info "查找trace报告..."
    
    local reports=($(ls -t *.nsys-rep 2>/dev/null | head -5))
    
    if [ ${#reports[@]} -eq 0 ]; then
        print_warning "未找到trace报告文件"
        return
    fi
    
    echo "找到以下trace报告:"
    for i in "${!reports[@]}"; do
        echo "  $((i+1)). ${reports[$i]}"
    done
    
    echo ""
    read -p "选择要查看的报告编号 (1-${#reports[@]}): " choice
    
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#reports[@]}" ]; then
        local selected_report="${reports[$((choice-1))]}"
        print_info "打开报告: $selected_report"
        
        # 尝试使用GUI打开
        if command -v nsys-ui &> /dev/null; then
            nsys-ui "$selected_report" &
        else
            print_warning "nsys-ui不可用，尝试导出为SQLite格式"
            local sqlite_name="${selected_report%.nsys-rep}.sqlite"
            nsys export --type sqlite --output "$sqlite_name" "$selected_report"
            print_success "已导出为: $sqlite_name"
        fi
    else
        print_error "无效的选择"
    fi
}

# 主函数
main() {
    print_info "SparseDrive Trace工具"
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 解析参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--test)
            run_test
            exit 0
            ;;
        -c|--config)
            show_config
            exit 0
            ;;
        -v|--view)
            view_reports
            exit 0
            ;;
        -s1|--stage1)
            run_stage1
            exit 0
            ;;
        -s2|--stage2)
            run_stage2
            exit 0
            ;;
        "")
            show_help
            exit 0
            ;;
        *)
            # 运行指定的脚本
            run_trace "$@"
            ;;
    esac
}

# 运行主函数
main "$@"
