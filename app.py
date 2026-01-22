import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- LÓGICA DE PROCESSAMENTO (A Mesma de Antes) ---

class ColorMatcher:
    def get_image_stats(self, image):
        # Converte para LAB
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        (l, a, b) = cv2.split(image_lab)
        return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())

    def find_best_reference(self, source_img, reference_images):
        src_stats = self.get_image_stats(source_img)
        src_l, src_a, src_b = src_stats[0], src_stats[2], src_stats[4]
        
        best_ref = None
        min_diff = float('inf')
        best_ref_name = ""
        
        for name, ref_img in reference_images.items():
            ref_stats = self.get_image_stats(ref_img)
            ref_l, ref_a, ref_b = ref_stats[0], ref_stats[2], ref_stats[4]
            
            # Distância Euclidiana (Luz + Cor)
            diff_l = (src_l - ref_l) ** 2
            diff_a = (src_a - ref_a) ** 2
            diff_b = (src_b - ref_b) ** 2
            
            # Pesos ajustáveis (dando leve prioridade para exposição)
            total_diff = np.sqrt((diff_l * 1.5) + diff_a + diff_b)
            
            if total_diff < min_diff:
                min_diff = total_diff
                best_ref = ref_img
                best_ref_name = name
                
        return best_ref, best_ref_name

    def apply_color_transfer(self, source, target, strength=0.85):
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        # Separa canais (Note os nomes: l_src, a_src, b_src)
        (l_src, a_src, b_src) = cv2.split(source_lab)
        (l_tar, a_tar, b_tar) = cv2.split(target_lab)

        l_mean_src, l_std_src = l_src.mean(), l_src.std()
        a_mean_src, a_std_src = a_src.mean(), a_src.std()
        b_mean_src, b_std_src = b_src.mean(), b_src.std()

        l_mean_tar, l_std_tar = l_tar.mean(), l_tar.std()
        a_mean_tar, a_std_tar = a_tar.mean(), a_tar.std()
        b_mean_tar, b_std_tar = b_tar.mean(), b_tar.std()

        eps = 1e-5
        
        # --- CORREÇÃO AQUI ---
        # Antes estava usando 'a' e 'b', agora está correto usando 'a_src' e 'b_src'
        l_new = ((l_src - l_mean_src) * (l_std_tar / (l_std_src + eps))) + l_mean_tar
        a_new = ((a_src - a_mean_src) * (a_std_tar / (a_std_src + eps))) + a_mean_tar
        b_new = ((b_src - b_mean_src) * (b_std_tar / (b_std_src + eps))) + b_mean_tar
        # ---------------------

        l_new = np.clip(l_new, 0, 255)
        a_new = np.clip(a_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        transfer_lab = cv2.merge([l_new, a_new, b_new])
        transfer_bgr = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
        
        if strength < 1.0:
            return cv2.addWeighted(transfer_bgr, strength, source, 1.0 - strength, 0)
        return transfer_bgr

# --- INTERFACE STREAMLIT ---

def load_image(image_file):
    img = Image.open(image_file)
    # Streamlit/PIL usa RGB, OpenCV usa BGR. Precisamos converter.
    img_array = np.array(img.convert('RGB'))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

st.set_page_config(page_title="Brand Color Corrector", layout="wide")

st.title("🎨 Corretor de Cores - Brand Guidelines")
st.markdown("Essa ferramenta ajusta automaticamente a cor da foto baseada nas referências do Brand.")

# --- BARRA LATERAL (REFERÊNCIAS) ---
st.sidebar.header("1. Referências do Brand")
st.sidebar.info("Faça upload das fotos 'perfeitas' que servirão de guia.")
ref_files = st.sidebar.file_uploader("Upload Referências (JPG/PNG)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

reference_images = {}
if ref_files:
    for ref_file in ref_files:
        # Carrega e salva na memória
        img = load_image(ref_file)
        # Redimensiona refs para processar mais rápido (não afeta a estatística de cor)
        img = cv2.resize(img, (300, 300)) 
        reference_images[ref_file.name] = img
    st.sidebar.success(f"{len(reference_images)} referências carregadas!")
else:
    st.sidebar.warning("Por favor, suba pelo menos uma imagem de referência.")

# --- ÁREA PRINCIPAL (INPUT USUÁRIO) ---
st.header("2. Foto para Corrigir")
target_file = st.file_uploader("Arraste a foto que precisa de correção", type=['png', 'jpg', 'jpeg'])

if target_file and reference_images:
    # Carregar imagem original
    input_img = load_image(target_file)
    
    # Botão de processar
    if st.button("✨ Corrigir Cores"):
        matcher = ColorMatcher()
        
        with st.spinner('Analisando luz e cor...'):
            # 1. Achar melhor referencia
            best_ref_img, best_ref_name = matcher.find_best_reference(input_img, reference_images)
            
            # 2. Aplicar correção
            corrected_img = matcher.apply_color_transfer(input_img, best_ref_img, strength=0.85)
            
            # Conversão para exibição
            input_rgb = bgr_to_rgb(input_img)
            corrected_rgb = bgr_to_rgb(corrected_img)
            ref_rgb = bgr_to_rgb(best_ref_img)

        # --- EXIBIÇÃO DE RESULTADOS ---
        st.success(f"Correção aplicada baseada na referência: **{best_ref_name}**")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.image(input_rgb, caption="Original", use_container_width=True)
        
        with col2:
            st.image(corrected_rgb, caption="Resultado Corrigido", use_container_width=True)
            
        with col3:
            st.image(ref_rgb, caption=f"Ref Usada ({best_ref_name})", use_container_width=True)

        # Botão de Download
        # Precisamos converter o array numpy de volta para bytes para o botão de download
        result_pil = Image.fromarray(corrected_rgb)
        
        # Salvar em buffer de memória
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG", quality=95)
        byte_im = buf.getvalue()

        st.download_button(
            label="⬇️ Baixar Imagem Corrigida",
            data=byte_im,
            file_name=f"corrected_{target_file.name}",
            mime="image/jpeg"
        )

elif target_file and not reference_images:
    st.error("Você precisa subir imagens de referência na barra lateral primeiro!")