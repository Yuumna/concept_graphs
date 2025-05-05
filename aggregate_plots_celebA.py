import os
import math
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


def find_pdfs(base_path):
    pdf_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def extract_first_image_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        return None
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def create_aggregate_image(image_title_pairs, output_path='aggregated_output.png', padding=20):
    font = ImageFont.load_default()

    num_images = len(image_title_pairs)
    grid_size = math.ceil(math.sqrt(num_images))

    img_width, img_height = image_title_pairs[0][0].width, image_title_pairs[0][0].height + 20
    total_width = grid_size * img_width + (grid_size + 1) * padding
    total_height = grid_size * img_height + (grid_size + 1) * padding

    final_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(final_img)

    for idx, (img, title) in enumerate(image_title_pairs):
        row = idx // grid_size
        col = idx % grid_size

        x = padding + col * (img_width + padding)
        y = padding + row * (img_height + padding)

        # Draw the title
        draw.text((x, y), title, fill='black', font=font)
        # Paste the image
        final_img.paste(img, (x, y + 20))

    final_img.save(output_path)
    print(f"Saved aggregated image to {output_path}")


def main(base_path):
    pdf_paths = find_pdfs(base_path)
    print(f"Found {len(pdf_paths)} PDF files.")

    image_title_pairs = []
    for pdf_path in pdf_paths:
        img = extract_first_image_from_pdf(pdf_path)
        if img is not None:
            rel_path = os.path.relpath(pdf_path, base_path)
            # Remove the last folder in the path for the title
            title = os.path.dirname(rel_path)
            image_title_pairs.append((img, title))
        else:
            print(f"Warning: No image extracted from {pdf_path}")

    if image_title_pairs:
        create_aggregate_image(image_title_pairs)
    else:
        print("No images to aggregate.")


if __name__ == "__main__":
    base_path = "results/output_all/celeba-3classes-smiling-10000_100"
    main(base_path)
