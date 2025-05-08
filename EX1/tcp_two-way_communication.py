import socket
import threading
import sys

# 配置参数
ADDR_RECV = ('127.0.0.1', 1028)  
ADDR_SEND = ('127.0.0.1', 1027)
# ADDR_RECV = ('127.0.0.1', 1027)  # 服务器端地址和端口
# ADDR_SEND = ('127.0.0.1', 1028)  # 客户端发送的目标地址
BUFFER_SIZE = 1024               # 缓冲区大小

# 创建TCP套接字
tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定并监听接收套接字（服务器端）
tcp_server.bind(ADDR_RECV)
tcp_server.listen(1)  # 监听最大连接数为1（可根据需要增加）

def listen():
    """监听接收消息的线程函数"""
    print(f"[系统] 开始在 {ADDR_RECV} 监听消息...")
    
    # 接受客户端连接
    conn, addr = tcp_server.accept()
    print(f"[系统] 与 {addr} 建立连接...")
    
    while True:
        try:
            # 接收消息
            recv_msg = conn.recv(BUFFER_SIZE)
            if not recv_msg:
                # 如果没有数据，退出
                break
                
            msg_decoded = recv_msg.decode('gbk')
            
            # 打印接收到的消息
            print(f"\n[来自 {addr} 的消息]: {msg_decoded}")
            
            # 如果收到退出指令则关闭连接
            if msg_decoded.upper() == "BYE":
                print("[系统] 收到退出指令，关闭连接...")
                conn.close()
                break
                
        except Exception as e:
            print(f"[错误] 接收消息时出错: {e}")
            break

def send_message():
    """发送消息的函数"""
    try:
        # 连接到服务器端
        tcp_client.connect(ADDR_SEND)
        print(f"[系统] 已连接到 {ADDR_SEND}，准备发送消息(输入'BYE'退出)")
        
        while True:
            # 获取用户输入
            info = input("[请输入消息]: ")
            
            # 发送消息
            tcp_client.send(info.encode('gbk'))
            
            # 检查退出指令
            if info.upper() == "BYE":
                print("[系统] 发送退出指令，程序结束...")
                break
    except KeyboardInterrupt:
        print("\n[系统] 用户中断，程序结束...")
    finally:
        # 清理资源
        tcp_client.close()
        sys.exit(0)

# 启动监听线程
listen_thread = threading.Thread(target=listen, daemon=True)
listen_thread.start()

# 主线程负责发送消息
send_message()
