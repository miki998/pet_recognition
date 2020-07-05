from src import *
import argparse




def main():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument(
        'action', choices=['add', 'modify', 'delete'], help='action of upstream nodes, one of add, modify, delete')
    parser.add_argument(
        '-n', '--name', required=True, type=str, help='upstream node name')
    parser.add_argument(
        '-l', '--loki_id', required=False, type=int, help='loki id')
    parser.add_argument(
        '-p', '--port', required=False, type=int, help='port')
    parser.add_argument(
        '-i', '--ip_hash', required=False, type=int, choices=[0, 1], help='ip_hash')
    parser.add_argument(
        '-o', '--online', required=False, type=int, choices=[0, 1], help='online')
    args = parser.parse_args()

    if args.action == "add":
        print args.name, args.loki_id, args.port, args.ip_hash, args.online
    elif args.action == "modify":
        _dict = {}
        for i in ["loki_id", "port", "ip_hash", "online"]:
            if getattr(args, i) is not None:
                _dict[i] = getattr(args, i)
        print args.name, _dict
    elif args.action == "delete":
        print args.name

if __name__ == '__main__':
	main()