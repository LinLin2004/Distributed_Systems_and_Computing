import socket
import threading
from datetime import datetime

def daytime_server(port=6666):
    """Daytime服务端"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('0.0.0.0', port))
        print(f"[服务端] 正在UDP端口 {port} 运行...")
        try:
            while True:
                _, addr = s.recvfrom(1024)
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S\r\n")
                s.sendto(time_str.encode('ascii'), addr)
                print(f"[服务端] 向 {addr} 发送时间")
        except KeyboardInterrupt:
            print("[服务端] 关闭")

def daytime_client(server='127.0.0.1', port=6666):
    """Daytime客户端"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(2)
        try:
            for i in range(3):  # 发送3次请求
                s.sendto(b'', (server, port))
                data, _ = s.recvfrom(1024)
                print(f"[客户端] 收到时间: {data.decode('ascii').strip()}")
        except Exception as e:
            print(f"[客户端] 错误: {e}")

if __name__ == '__main__':
    # 启动服务端线程
    server_thread = threading.Thread(target=daytime_server, daemon=True)
    server_thread.start()
    
    # 启动客户端
    daytime_client()
    
    # 等待客户端完成
    server_thread.join(timeout=5)
    print("测试完成")