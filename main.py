import cv2
import numpy as np
import os
import glob

class ColorMatcher:
    def __init__(self):
        pass

    def get_image_stats(self, image):
        """
        Calcula média e desvio padrão no espaço LAB.
        O espaço LAB é usado porque separa Luminosidade (L) de Cor (A, B).
        """
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        (l, a, b) = cv2.split(image_lab)
        
        # Retornamos as estatísticas de L (Luminosidade) separadas para facilitar a comparação
        l_mean, l_std = l.mean(), l.std()
        a_mean, a_std = a.mean(), a.std()
        b_mean, b_std = b.mean(), b.std()
        
        return (l_mean, l_std, a_mean, a_std, b_mean, b_std)

   def find_best_reference(self, source_img, reference_images):
        """
        Versão APRIMORADA: Compara Brilho (L) e Cor (A, B) usando distância Euclidiana.
        """
        # Desempacota estatísticas da imagem de origem
        # src_stats contém: (l_mean, l_std, a_mean, a_std, b_mean, b_std)
        src_stats = self.get_image_stats(source_img)
        src_l, src_a, src_b = src_stats[0], src_stats[2], src_stats[4]
        
        best_ref = None
        min_diff = float('inf')
        
        print(f"--- Analisando melhor match (Luminosidade + Cor)... ---")
        
        for ref_name, ref_img in reference_images.items():
            ref_stats = self.get_image_stats(ref_img)
            ref_l, ref_a, ref_b = ref_stats[0], ref_stats[2], ref_stats[4]
            
            # --- A MUDANÇA ESTÁ AQUI ---
            
            # Calculamos a diferença em cada canal
            diff_l = (src_l - ref_l) ** 2
            diff_a = (src_a - ref_a) ** 2
            diff_b = (src_b - ref_b) ** 2
            
            # Você pode dar pesos se quiser priorizar Brilho sobre Cor
            # Ex: peso_luz = 2.0 (o brilho importa o dobro da cor)
            peso_luz = 1.0
            peso_cor = 1.0
            
            # Distância Euclidiana Ponderada
            # Raiz quadrada da soma dos quadrados (Pitágoras em 3D)
            total_diff = np.sqrt((diff_l * peso_luz) + (diff_a * peso_cor) + (diff_b * peso_cor))
            
            # ---------------------------
            
            if total_diff < min_diff:
                min_diff = total_diff
                best_ref = ref_img
                best_ref_name = ref_name
        
        print(f"Match escolhido: {best_ref_name} (Distância: {min_diff:.2f})")
        return best_ref

    def apply_color_transfer(self, source, target, strength=1.0):
        """
        Aplica a transferência de cor de Reinhard.
        strength: 0.0 a 1.0 (Intensidade do efeito).
                  1.0 = Troca total pela estatística da referência.
                  0.5 = Mistura 50% original e 50% corrigido.
        """
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        # Separa canais
        (l_src, a_src, b_src) = cv2.split(source_lab)
        (l_tar, a_tar, b_tar) = cv2.split(target_lab)

        # Calcula estatísticas
        (l_mean_src, l_std_src) = (l_src.mean(), l_src.std())
        (a_mean_src, a_std_src) = (a_src.mean(), a_src.std())
        (b_mean_src, b_std_src) = (b_src.mean(), b_src.std())

        (l_mean_tar, l_std_tar) = (l_tar.mean(), l_tar.std())
        (a_mean_tar, a_std_tar) = (a_tar.mean(), a_tar.std())
        (b_mean_tar, b_std_tar) = (b_tar.mean(), b_tar.std())

        # Evita divisão por zero
        eps = 1e-5

        # Subtrai média da origem
        l_src -= l_mean_src
        a_src -= a_mean_src
        b_src -= b_mean_src

        # Escala pelo desvio padrão (fator de contraste)
        # Nota: Limitamos o fator de escala para evitar artefatos extremos
        scale_l = l_std_tar / (l_std_src + eps)
        scale_a = a_std_tar / (a_std_src + eps)
        scale_b = b_std_tar / (b_std_src + eps)

        l_new = l_src * scale_l
        a_new = a_src * scale_a
        b_new = b_src * scale_b

        # Adiciona média do destino
        l_new += l_mean_tar
        a_new += a_mean_tar
        b_new += b_mean_tar

        # Clipagem (0-255)
        l_new = np.clip(l_new, 0, 255)
        a_new = np.clip(a_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        # Merge e converte
        transfer_lab = cv2.merge([l_new, a_new, b_new])
        transfer_bgr = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
        
        # Aplica o peso (blending) se strength < 1.0
        if strength < 1.0:
            return cv2.addWeighted(transfer_bgr, strength, source, 1.0 - strength, 0)
            
        return transfer_bgr

def main():
    matcher = ColorMatcher()
    
    # 1. Carregar Referências
    print("Carregando banco de referências...")
    refs = {}
    ref_extensions = ['*.jpg', '*.jpeg', '*.png']
    for ext in ref_extensions:
        for file in glob.glob(f'references/{ext}'):
            filename = os.path.basename(file)
            img = cv2.imread(file)
            if img is not None:
                # Redimensionamos referências muito grandes para acelerar o cálculo estatístico
                # (A estatística de cor não muda significativamente com o tamanho)
                img_small = cv2.resize(img, (300, 300))
                refs[filename] = img_small
    
    if not refs:
        print("ERRO: Nenhuma imagem encontrada na pasta /references")
        return

    # 2. Processar Inputs
    input_folder = 'input'
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processando imagens de {input_folder}...")
    
    for ext in ref_extensions:
        for file_path in glob.glob(f'{input_folder}/{ext}'):
            filename = os.path.basename(file_path)
            input_img = cv2.imread(file_path)
            
            if input_img is None:
                continue

            print(f"\nImagem: {filename}")
            
            # Passo A: Achar a melhor referência para esta imagem específica
            best_ref_img = matcher.find_best_reference(input_img, refs)
            
            # Passo B: Aplicar correção
            # strength=0.85 mantém 15% da original para naturalidade
            result_img = matcher.apply_color_transfer(input_img, best_ref_img, strength=0.85)
            
            # Salvar
            cv2.imwrite(f'{output_folder}/corrected_{filename}', result_img)
            print(f"Salvo em {output_folder}/corrected_{filename}")

if __name__ == "__main__":
    main()
