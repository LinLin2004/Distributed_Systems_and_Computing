import socket
import threading
import sys

# 配置参数
ADDR_RECV = ('127.0.0.1', 1025)  # 接收消息的地址
ADDR_SEND = ('127.0.0.1', 1026)  # 发送消息的目标地址
BUFFER_SIZE = 1024               # 缓冲区大小

# 创建UDP套接字
udp_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定接收套接字
udp_recv.bind(ADDR_RECV)

def listen():
    """监听接收消息的线程函数"""
    print(f"[系统] 开始在 {ADDR_RECV} 监听消息...")
    while True:
        try:
            # 接收消息
            recv_msg, recv_addr = udp_recv.recvfrom(BUFFER_SIZE)
            msg_decoded = recv_msg.decode('gbk')
            
            # 打印接收到的消息
            print(f"\n[来自 {recv_addr} 的消息]: {msg_decoded}")
            
            # 如果收到退出指令则关闭
            if msg_decoded.upper() == "BYE":
                print("[系统] 收到退出指令，关闭接收...")
                udp_recv.close()
                break
                
        except OSError:
            # 套接字关闭时的正常退出
            break
        except Exception as e:
            print(f"[错误] 接收消息时出错: {e}")
            break

# 启动监听线程
listen_thread = threading.Thread(target=listen, daemon=True)
listen_thread.start()

# 主线程处理消息发送
try:
    print(f"[系统] 准备向 {ADDR_SEND} 发送消息(输入'BYE'退出)")
    while True:
        # 获取用户输入
        info = input("[请输入消息]: ")
        
        # 发送消息
        udp_send.sendto(info.encode('gbk'), ADDR_SEND)
        
        # 检查退出指令
        if info.upper() == "BYE" :
            print("[系统] 发送退出指令，程序结束...")
            break
            
except KeyboardInterrupt:
    print("\n[系统] 用户中断，程序结束...")
finally:
    # 清理资源
    udp_recv.close()
    udp_send.close()
    sys.exit(0)