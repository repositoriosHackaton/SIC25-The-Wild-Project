# **WildPassPro - Suite de Seguridad**

**WildPassPro** es una aplicación avanzada de seguridad diseñada para gestionar, analizar y proteger contraseñas y credenciales. Combina inteligencia artificial, técnicas de criptografía y análisis de vulnerabilidades para ofrecer una solución integral en la gestión de credenciales.

---

## **Características Principales**

### **🛠️ Generadores**
- **🔑 Generador de Contraseñas Seguras**: Crea contraseñas robustas con longitud personalizable (12-32 caracteres).
- **🔑 Generador de Llaves de Acceso**: Genera llaves de acceso únicas y seguras para APIs o sistemas.

### **🔒 Bóveda de Contraseñas**
- **Almacenamiento Seguro**: Guarda contraseñas cifradas con **Fernet** (AES-128).
- **Gestión de Credenciales**: Añade, visualiza y elimina credenciales de forma segura.
- **Cifrado Automático**: Los datos se cifran automáticamente al guardarse.

### **🔍 Analizador de Contraseñas**
- **Detección de Debilidades**: Identifica contraseñas débiles basadas en patrones comunes.
- **Red Neuronal**: Clasifica contraseñas en **Débil**, **Media** o **Fuerte** con un modelo entrenado.
- **Análisis de Groq**: Usa **Llama3-70b** para un análisis detallado de la seguridad de la contraseña.

### **💬 Asistente de Seguridad**
- **Chatbot Inteligente**: Responde preguntas sobre seguridad, mejores prácticas y gestión de credenciales.
- **Integración con Groq**: Proporciona respuestas precisas y contextuales.

### **🌐 Escáner de Vulnerabilidades Web**
- **Detección de Vulnerabilidades**: Escanea sitios web en busca de **XSS**, **SQL Injection** y **CSRF**.
- **Explicación Detallada**: Usa Groq para explicar las vulnerabilidades encontradas y cómo solucionarlas.

### **🔐 Verificador de Fugas de Datos**
- **Comprobación de Fugas**: Verifica si una contraseña ha sido expuesta en fugas de datos conocidas usando la API de **Have I Been Pwned**.

---

## **Tecnologías Utilizadas**

- **Inteligencia Artificial**:
  - **Groq API** con el modelo **Llama3-70b** para análisis avanzado y chat bot.
  - **Red Neuronal** entrenada con TensorFlow/Keras para clasificación de contraseñas.
  
- **Criptografía**:
  - **Fernet (AES-128)** para cifrado seguro de contraseñas.
  - **SHA-1** para verificación de fugas de datos.

- **Frameworks y Librerías**:
  - **Streamlit** para la interfaz de usuario.
  - **Pandas** y **NumPy** para manejo de datos.
  - **Scikit-learn** para preprocesamiento.
  - **Requests** para interacción con APIs externas.

---

## **Instalación y Uso**

### **Requisitos**
- Python 3.9 o superior.
- Librerías necesarias: `streamlit`, `tensorflow`, `pandas`, `numpy`, `cryptography`, `requests`, `scikit-learn`.

### **Instalación**
1. Clona el repositorio:
   ```bash
   git clone https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project.git
   cd PROYECTO-IA-SIC-The-Wild-Project
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

### **Uso**
1. **Genera contraseñas seguras** en la pestaña **🛠️ Generadores**.
2. **Almacena y gestiona** tus credenciales en la **🔒 Bóveda**.
3. **Analiza contraseñas** existentes en la pestaña **🔍 Analizador**.
4. **Escanea sitios web** en busca de vulnerabilidades en **🌐 Escáner Web**.
5. **Consulta al asistente** de seguridad en **💬 Chatbot**.
6. **Verifica fugas de datos** en **🔐 Verificador de Fugas**.

---

## **Arquitectura del Sistema**

### **Red Neuronal**
- **Capas**:
  - **Capa Oculta 1**: 64 neuronas con activación **ReLU**.
  - **Capa Oculta 2**: 32 neuronas con activación **ReLU**.
  - **Capa Oculta 3**: 16 neuronas con activación **ReLU**.
  - **Capa de Salida**: 3 neuronas con activación **Softmax** (clasificación en 3 clases).
- **Entrenamiento**:
  - **Dataset**: 14,501 contraseñas etiquetadas.
  - **Optimizador**: Adam con tasa de aprendizaje adaptativa.
  - **Función de Pérdida**: `sparse_categorical_crossentropy`.
  - **Regularización**: Dropout y BatchNormalization para evitar sobreajuste.

### **Cifrado**
- **Fernet (AES-128)**:
  - Genera una clave de cifrado única al iniciar la aplicación.
  - Cifra y descifra archivos automáticamente.

### **Integración con APIs**
- **Groq API**: Para análisis avanzado y respuestas contextuales.
- **Have I Been Pwned API**: Para verificación de fugas de datos.

---

## **Seguridad**
- **Cifrado de Datos**: Todas las contraseñas se almacenan cifradas.
- **Protección de Acceso**: Requiere una contraseña maestra para acceder a la bóveda.
- **Verificación de Fugas**: Comprueba si las contraseñas han sido expuestas en fugas de datos.

---

## **Competencia**
Las mejoras de WildPassPro vs la competencia:
![image](https://github.com/user-attachments/assets/30e3cc31-d06c-4cc3-ac0a-b612e48c5d22)

Matriz competitiva de WildPassPro:
![image](https://github.com/user-attachments/assets/65a45f8c-5d8c-43d4-a6b0-092fb6b22d6c)

---

## **Graficas**
![image](https://github.com/user-attachments/assets/724f57d2-8b8e-40ad-b26f-fc1a571f562e)
![image](https://github.com/user-attachments/assets/e399d5a4-1449-4b43-8455-218c67e67961)
![image](https://github.com/user-attachments/assets/99d55c90-c5d8-4b2e-960d-c9f479dea4c9)
![image](https://github.com/user-attachments/assets/cdcfc2cb-a989-49da-afa6-20a3bde1ad37)
![image](https://github.com/user-attachments/assets/361edd0c-3865-490b-8405-04b0b3cccb0c)
![image](https://github.com/user-attachments/assets/1af7fc57-e302-4b54-9b0c-7f74debf763d)
![image](https://github.com/user-attachments/assets/d7736a4c-c143-4e06-bc37-c71720c505bb)
![image](https://github.com/user-attachments/assets/b89f8f5b-20b1-4c53-ae44-40a0978a8d63)
![image](https://github.com/user-attachments/assets/be1c3ed6-435a-4ec8-9227-400585e51513)
![image](https://github.com/user-attachments/assets/14659804-3b39-48f5-b64c-b09205ae1885)
![image](https://github.com/user-attachments/assets/c4d62e50-2549-4926-bf16-af9f5237f7a1)
![image](https://github.com/user-attachments/assets/30197a83-fa0a-45bb-ab0d-22233575b0d2)


---

## **Contribuciones**
¡Las contribuciones son bienvenidas! Si deseas mejorar el proyecto, sigue estos pasos:
1. Haz un fork del repositorio.
2. Crea una rama con tu nueva funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Haz push a la rama (`git push origin nueva-funcionalidad`).
5. Abre un Pull Request.

---

## **Contacto**
Autores
- AndersonP444 (Andersonjperdomo@gmail.com)
- DiegoAlviarez (dilanalviarez@gmail.com)
- Jeremyvr28 (jeremyvicent28@gmail.com)
- mrgrey28 (greymelmoreno@gmail.com)
- Kev1nM4nu (kenken29815793@gmail.com)


---

**WildPassPro** es una herramienta poderosa para proteger tus credenciales y mejorar tu seguridad en línea. ¡Pruébala y mantén tus datos seguros! 🔐
