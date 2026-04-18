import os
import easyocr
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import time
from tqdm import tqdm

# 创建输出目录
os.makedirs('./output/images', exist_ok=True)
os.makedirs('./output/text', exist_ok=True)


def pdf_to_images(pdf_path, output_folder='./output/images', dpi=300):
    """将PDF转换为图片"""
    print(f"正在将PDF转换为图片，DPI设置为{dpi}...")
    try:
        # 尝试直接转换PDF
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"转换PDF时出错: {e}")
        print("尝试使用PyPDF2分割PDF后再转换...")

        # 使用PyPDF2分割PDF为单页
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

            for i in range(num_pages):
                writer = PyPDF2.PdfWriter()
                writer.add_page(reader.pages[i])

                # 保存单页PDF
                single_pdf_path = f'./output/temp_page_{i + 1}.pdf'
                with open(single_pdf_path, 'wb') as single_pdf:
                    writer.write(single_pdf)

                # 转换单页PDF为图片
                try:
                    page_images = convert_from_path(single_pdf_path, dpi=dpi)
                    for j, img in enumerate(page_images):
                        img_path = os.path.join(output_folder, f'page_{i + 1}.jpg')
                        img.save(img_path, 'JPEG')

                    # 删除临时文件
                    os.remove(single_pdf_path)
                except Exception as e:
                    print(f"转换第{i + 1}页时出错: {e}")

        # 重新读取生成的图片
        pages = []
        for i in range(1, num_pages + 1):
            img_path = os.path.join(output_folder, f'page_{i}.jpg')
            if os.path.exists(img_path):
                pages.append(Image.open(img_path))

    # 保存图片
    image_paths = []
    for i, page in enumerate(tqdm(pages, desc="保存图片")):
        image_path = os.path.join(output_folder, f'page_{i + 1}.jpg')
        page.save(image_path, 'JPEG')
        image_paths.append(image_path)

    print(f"已成功将PDF转换为{len(image_paths)}张图片")
    return image_paths


def ocr_images(image_paths, output_text='./output/text/ocr_result.txt', language='ch_sim'):
    """对图片进行OCR识别并保存结果"""
    print(f"正在初始化OCR引擎，语言设置为: {language}")

    # 初始化EasyOCR阅读器
    reader = easyocr.Reader([language])

    # 打开输出文件
    with open(output_text, 'w', encoding='utf-8') as f:
        # 遍历所有图片
        for i, image_path in enumerate(tqdm(image_paths, desc="OCR识别")):
            try:
                # 进行OCR识别
                result = reader.readtext(image_path, detail=0)

                # 写入结果
                f.write(f"===== 第{i + 1}页 =====\n")
                for line in result:
                    f.write(line + '\n')
                f.write('\n\n')

                # 防止API调用过快
                time.sleep(0.5)

            except Exception as e:
                print(f"处理第{i + 1}页时出错: {e}")
                f.write(f"===== 第{i + 1}页 (处理出错) =====\n")
                f.write(f"错误信息: {str(e)}\n\n")

    print(f"OCR识别完成，结果已保存至: {output_text}")


def main(pdf_path, language='ch_sim', dpi=300):
    """主函数：协调PDF转换和OCR识别"""
    print(f"开始处理PDF文件: {pdf_path}")

    # 1. 转换PDF为图片
    image_paths = pdf_to_images(pdf_path, dpi=dpi)

    # 2. 对图片进行OCR识别
    output_text = f'./output/text/ocr_result_{os.path.basename(pdf_path)}.txt'
    ocr_images(image_paths, output_text, language)

    print("=" * 50)
    print("所有处理完成！")


if __name__ == "__main__":
    # 配置参数
    PDF_PATH = "../RAG/舒缓敏感性皮肤症状对策研究现状.pdf"  # 替换为你的PDF文件路径
    LANGUAGE = 'ch_sim'  # 语言代码，中文简体
    DPI = 300  # 图片分辨率，影响OCR识别效果

    # 检查文件是否存在
    if not os.path.exists(PDF_PATH):
        print(f"错误：文件不存在 - {PDF_PATH}")
    else:
        main(PDF_PATH, LANGUAGE, DPI)