import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const PlantixClone = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const navbar = document.querySelector('.navbar');
      if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="navbar fixed top-0 w-full bg-white shadow-md z-50 transition-all duration-300">
        <div className="max-w-6xl mx-auto px-4 md:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <img 
                src="https://placeholder-image-service.onrender.com/image/150x40?prompt=Plantix%20logo%20with%20green%20leaf%20design&id=logo1" 
                alt="Plantix logo con diseño de hoja verde"
                className="h-8 md:h-10"
              />
            </div>
            
            {/* Desktop Menu */}
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-green-800 hover:text-green-600 font-medium">Funciones</a>
              <a href="#library" className="text-green-800 hover:text-green-600 font-medium">Biblioteca</a>
              <a href="#community" className="text-green-800 hover:text-green-600 font-medium">Comunidad</a>
              <a href="#pricing" className="text-green-800 hover:text-green-600 font-medium">Precios</a>
              <button className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg">
                Descargar
              </button>
            </div>

            {/* Mobile Menu Button */}
            <button 
              className="md:hidden text-green-800"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>

          {/* Mobile Menu */}
          {isMenuOpen && (
            <div className="md:hidden mt-4 bg-white rounded-lg shadow-lg p-4">
              <a href="#features" className="block py-2 text-green-800 hover:text-green-600">Funciones</a>
              <a href="#library" className="block py-2 text-green-800 hover:text-green-600">Biblioteca</a>
              <a href="#community" className="block py-2 text-green-800 hover:text-green-600">Comunidad</a>
              <a href="#pricing" className="block py-2 text-green-800 hover:text-green-600">Precios</a>
              <button className="w-full bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg mt-2">
                Descargar
              </button>
            </div>
          )}
        </div>
      </nav>

      {/* Add margin for fixed nav */}
      <div className="pt-20">
      {/* Hero Section */}
      <header className="bg-green-50 py-12 px-4 md:px-8">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-3xl md:text-4xl font-bold text-green-800 mb-4">
            La mejor aplicación GRATIS para diagnosticar y tratar cultivos
          </h1>
          <p className="text-lg md:text-xl text-green-700 max-w-3xl mx-auto mb-8">
            Plantix ayuda a los agricultores a diagnosticar y tratar los problemas de sus cultivos, 
            mejorar la productividad y brinda conocimientos agrícolas. Alcance sus objetivos y 
            mejore su experiencia en la agricultura con Plantix.
          </p>
          <div className="bg-white rounded-lg shadow-md p-6 max-w-2xl mx-auto">
            <p className="text-green-800 font-medium">
              Con la confianza de la mayor comunidad agrícola
            </p>
          </div>
        </div>
      </header>

      {/* Features Section with animations */}
      <section id="features" className="py-16 px-4 md:px-8">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-2xl md:text-3xl font-bold text-green-800 text-center mb-12">
            Impulse la producción de sus cultivos
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {/* Feature 1 with animation */}
            <motion.div 
              className="bg-green-50 rounded-lg p-6 text-center cursor-pointer hover:shadow-lg transition-shadow duration-300"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="mb-4">
                <img 
                  src="https://placeholder-image-service.onrender.com/image/200x200?prompt=Smartphone%20showing%20a%20plant%20diagnosis%20app%20interface&id=plantix1" 
                  alt="Aplicación móvil mostrando diagnóstico de plantas con interfaz moderna"
                  className="mx-auto rounded-lg"
                />
              </div>
              <h3 className="text-xl font-semibold text-green-800 mb-3">
                Diagnostique la enfermedad de su cultivo
              </h3>
              <p className="text-green-700">
                Tome una foto de su cultivo enfermo y obtenga un diagnóstico gratuito y 
                sugerencias de tratamiento ¡en solo unos segundos!
              </p>
            </motion.div>
            
            {/* Feature 2 with animation */}
            <motion.div 
              className="bg-green-50 rounded-lg p-6 text-center cursor-pointer hover:shadow-lg transition-shadow duration-300"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="mb-4">
                <img 
                  src="https://placeholder-image-service.onrender.com/image/200x200?prompt=Agricultural%20experts%20discussing%20plants%20in%20a%20field&id=plantix2" 
                  alt="Expertos agrícolas discutiendo sobre cultivos en un campo"
                  className="mx-auto rounded-lg"
                />
              </div>
              <h3 className="text-xl font-semibold text-green-800 mb-3">
                Obtenga consejos de los expertos
              </h3>
              <p className="text-green-700">
                ¿Tiene alguna pregunta? No se preocupe. Nuestra comunidad de agroexpertos 
                le ayudará. También puede aprender sobre cultivos y ayudar a otros agricultores con su experiencia.
              </p>
            </motion.div>
            
            {/* Feature 3 with animation */}
            <motion.div 
              className="bg-green-50 rounded-lg p-6 text-center cursor-pointer hover:shadow-lg transition-shadow duration-300"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="mb-4">
                <img 
                  src="https://placeholder-image-service.onrender.com/image/200x200?prompt=Digital%20library%20with%20agricultural%20books%20and%20resources&id=plantix3" 
                  alt="Biblioteca digital con recursos agrícolas y libros sobre cultivos"
                  className="mx-auto rounded-lg"
                />
              </div>
              <h3 className="text-xl font-semibold text-green-800 mb-3">
                ¿Quiere maximizar la producción de sus cultivos?
              </h3>
              <p className="text-green-700">
                Nuestra biblioteca tiene todo lo que necesita. Con información sobre las 
                enfermedades específicas de su cultivo y los métodos de prevención, puede 
                garantizar el éxito de su cosecha.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section with animations */}
      <section className="py-16 bg-green-100 px-4 md:px-8">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-2xl md:text-3xl font-bold text-green-800 mb-8">
            Plantix en números
          </h2>
          <p className="text-lg text-green-700 max-w-3xl mx-auto mb-12">
            Plantix, la aplicación agrotécnica más descargada del mundo, ha respondido a 
            más de 100 millones de preguntas de los agricultores relacionadas con sus cultivos.
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <motion.div 
              className="bg-white rounded-lg p-4 shadow-sm"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
            >
              <div className="text-3xl font-bold text-green-800 mb-2">30M+</div>
              <div className="text-green-700">Descargas</div>
            </motion.div>
            <motion.div 
              className="bg-white rounded-lg p-4 shadow-sm"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <div className="text-3xl font-bold text-green-800 mb-2">100M+</div>
              <div className="text-green-700">Preguntas respondidas</div>
            </motion.div>
            <motion.div 
              className="bg-white rounded-lg p-4 shadow-sm"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              <div className="text-3xl font-bold text-green-800 mb-2">15+</div>
              <div className="text-green-700">Idiomas</div>
            </motion.div>
            <motion.div 
              className="bg-white rounded-lg p-4 shadow-sm"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.3 }}
            >
              <div className="text-3xl font-bold text-green-800 mb-2">50+</div>
              <div className="text-green-700">Países</div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Library Section with animated cards */}
      <section id="library" className="py-16 px-4 md:px-8 bg-white">
        <div className="max-w-6xl mx-auto">
          <motion.h2 
            className="text-2xl md:text-3xl font-bold text-green-800 text-center mb-8"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            Enfermedades de Plantas y Tratamientos
          </motion.h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {/* Library Card 1 */}
            <motion.div 
              className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-shadow duration-300 cursor-pointer"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
              whileHover={{ scale: 1.02 }}
            >
              <img 
                src="https://placeholder-image-service.onrender.com/image/300x200?prompt=Tomato%20plant%20with%20disease%20spots%20on%20leaves&id=library1" 
                alt="Planta de tomate con manchas de enfermedad en las hojas"
                className="w-full h-48 object-cover"
              />
              <div className="p-4">
                <h3 className="text-lg font-semibold text-green-800 mb-2">Tomate</h3>
                <p className="text-green-600 text-sm">15 enfermedades comunes</p>
              </div>
            </motion.div>

            {/* Library Card 2 */}
            <motion.div 
              className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-shadow duration-300 cursor-pointer"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <img 
                src="https://placeholder-image-service.onrender.com/image/300x200?prompt=Maize%20plant%20with%20healthy%20ears&id=library2" 
                alt="Planta de maíz con mazorcas saludables"
                className="w-full h-48 object-cover"
              />
              <div className="p-4">
                <h3 className="text-lg font-semibold text-green-800 mb-2">Maíz</h3>
                <p className="text-green-600 text-sm">12 enfermedades comunes</p>
              </div>
            </motion.div>

            {/* Library Card 3 */}
            <motion.div 
              className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-shadow duration-300 cursor-pointer"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.2 }}
              whileHover={{ scale: 1.02 }}
            >
              <img 
                src="https://placeholder-image-service.onrender.com/image/300x200?prompt=Potato%20plant%20with%20tuber%20diseases&id=library3" 
                alt="Planta de papa con enfermedades en tubérculos"
                className="w-full h-48 object-cover"
              />
              <div className="p-4">
                <h3 className="text-lg font-semibold text-green-800 mb-2">Papa</h3>
                <p className="text-green-600 text-sm">18 enfermedades comunes</p>
              </div>
            </motion.div>
          </div>

          <div className="text-center">
            <motion.button 
              className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-lg font-medium"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Ver todas las enfermedades
            </motion.button>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-16 px-4 md:px-8 bg-green-50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-2xl md:text-3xl font-bold text-green-800 text-center mb-12">
            Vea lo que dicen nuestros usuarios
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {/* Testimonial 1 with animation */}
            <motion.div 
              className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300"
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              whileHover={{ y: -5 }}
            >
              <p className="text-green-700 mb-4 italic">
                "La aplicación es eficiente y fácil de usar, lo que hace que identificar 
                las enfermedades de los cultivos y buscar tratamientos químicos y biológicos 
                sea pan comido."
              </p>
              <div className="flex items-center">
                <img 
                  src="https://placeholder-image-service.onrender.com/image/60x60?prompt=Brazilian%20farmer%20smiling%20in%20a%20field&id=testimonial1" 
                  alt="José Souza, agricultor brasileño sonriendo en un campo"
                  className="w-12 h-12 rounded-full mr-3"
                />
                <div>
                  <div className="font-semibold text-green-800">José Souza</div>
                  <div className="text-green-600">Agricultor, Brasil</div>
                </div>
              </div>
            </motion.div>
            
            {/* Testimonial 2 with animation */}
            <motion.div 
              className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
              whileHover={{ y: -5 }}
            >
              <p className="text-green-700 mb-4 italic">
                "Como agrónomo, recomiendo mucho esta aplicación. Ha sido eficaz en la 
                identificación y para proporcionar soluciones para combatir las enfermedades 
                de las plantas."
              </p>
              <div className="flex items-center">
                <img 
                  src="https://placeholder-image-service.onrender.com/image/60x60?prompt=Spanish%20agronomist%20with%20notebook%20in%20field&id=testimonial2" 
                  alt="Alejandro Escarra, agrónomo español con cuaderno en el campo"
                  className="w-12 h-12 rounded-full mr-3"
                />
                <div>
                  <div className="font-semibold text-green-800">Alejandro Escarra</div>
                  <div className="text-green-600">Agrónomo | España</div>
                </div>
              </div>
            </motion.div>
            
            {/* Testimonial 3 with animation */}
            <motion.div 
              className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300"
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              whileHover={{ y: -5 }}
            >
              <p className="text-green-700 mb-4 italic">
                "Esta aplicación me ha proporcionado excelentes análisis y soluciones para 
                las enfermedades de mis plantas. ¡La recomiendo encarecidamente a cualquiera 
                que desee mejorar la salud de sus cultivos!"
              </p>
              <div className="flex items-center">
                <img 
                  src="https://placeholder-image-service.onrender.com/image/60x60?prompt=Indonesian%20female%20farmer%20with%20healthy%20plants&id=testimonial3" 
                  alt="Wati Singarimbun, agricultora indonesia con plantas saludables"
                  className="w-12 h-12 rounded-full mr-3"
                />
                <div>
                  <div className="font-semibold text-green-800">Wati Singarimbun</div>
                  <div className="text-green-600">Agricultora | Indonesia</div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      

      <footer className="bg-green-900 text-white py-12 px-4 md:px-8">
        <div className="max-w-6xl mx-auto text-center">
          <div className="mb-8">
            <h3 className="text-2xl font-bold mb-4">Descarga Plantix ahora</h3>
            <div className="flex justify-center space-x-4">
              <button className="bg-green-700 hover:bg-green-600 text-white px-6 py-3 rounded-lg">
                App Store
              </button>
              <button className="bg-green-700 hover:bg-green-600 text-white px-6 py-3 rounded-lg">
                Google Play
              </button>
            </div>
          </div>
          <p className="text-green-200">
            © 2023 Plantix. Todos los derechos reservados.
          </p>
        </div>
      </footer>
    </div>
    </div>
  );
};

export default PlantixClone;
