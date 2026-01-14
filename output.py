from pipeline import create_paper_homepage, create_paper_explanation
import sys

def main():
    """主函数：让用户选择生成类型"""
    print("📚 论文处理工具")
    print("=" * 40)
    print("请选择要生成的类型：")
    print("1. 论文主页（HTML格式）")
    print("2. 论文讲解推文（Markdown格式）")
    print("3. 退出")
    print("=" * 40)
    
    while True:
        try:
            choice = input("请输入选项 (1/2/3): ").strip()
            
            if choice == "1":
                # 生成论文主页
                print("\n🏠 选择：生成论文主页")
                
                # 输入参数
                pdf_path = input("请输入PDF文件路径 [默认: ./pdf/Human-Agent.pdf]: ").strip()
                if not pdf_path:
                    pdf_path = "./pdf/Human-Agent.pdf"
                
                template_name = input("请输入模板文件名 [默认: ./muban_clean/purplepaper.html]: ").strip()
                if not template_name:
                    template_name = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/muban_clean/purplepaper.html"
                
                output_dir = input("请输入输出目录 [默认: ./html_output/]: ").strip()
                if not output_dir:
                    output_dir = "./html_output/"
                
                print(f"\n🚀 开始生成论文主页...")
                print(f"   - PDF: {pdf_path}")
                print(f"   - 模板: {template_name}")
                print(f"   - 输出目录: {output_dir}")
                
                try:
                    html_path = create_paper_homepage(
                        pdf_path=pdf_path,
                        template_name=template_name,
                        output_dir=output_dir
                    )
                    print(f"\n✅ 主页已生成: {html_path}")
                    
                    # 询问是否打开浏览器
                    open_browser = input("是否在浏览器中打开? (y/n): ").lower()
                    if open_browser == 'y':
                        import webbrowser
                        webbrowser.open(f"file://{html_path}")
                
                except Exception as e:
                    print(f"❌ 生成失败: {e}")
                
                break
                
            elif choice == "2":
                # 生成论文讲解推文
                print("\n🐦 选择：生成论文讲解推文")
                
                # 输入参数
                pdf_path = input("请输入PDF文件路径 [默认: ./pdf/Human-Agent.pdf]: ").strip()
                if not pdf_path:
                    pdf_path = "./pdf/Human-Agent.pdf"
                
                output_dir = input("请输入输出目录 [默认: ./tweet_output/]: ").strip()
                if not output_dir:
                    output_dir = "./tweet_output/"
                
                print(f"\n🚀 开始生成论文讲解推文...")
                print(f"   - PDF: {pdf_path}")
                print(f"   - 输出目录: {output_dir}")
                
                try:
                    md_path = create_paper_explanation(
                        pdf_path=pdf_path,
                        output_dir=output_dir
                    )
                    print(f"\n✅ 论文讲解推文已生成: {md_path}")
                    
                    # 显示部分内容预览
                    try:
                        with open(md_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.split('\n')
                            print("\n📄 内容预览:")
                            print("-" * 40)
                            for i in range(min(10, len(lines))):
                                print(lines[i])
                            if len(lines) > 10:
                                print("...")
                            print("-" * 40)
                    except:
                        pass
                
                except Exception as e:
                    print(f"❌ 生成失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                break
                
            elif choice == "3":
                print("👋 退出程序")
                sys.exit(0)
                
            else:
                print("❌ 无效选项，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()