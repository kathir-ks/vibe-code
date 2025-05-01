# Dockerfile

# Use the official Nginx image as a base
FROM nginx:stable-alpine

# Remove the default Nginx welcome page (optional)
RUN rm -f /usr/share/nginx/html/index.html /usr/share/nginx/html/50x.html

# Copy your static application files to the Nginx web root directory
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY script.js /usr/share/nginx/html/

# Copy your custom Nginx configuration
# If you named your config default.conf, copy it here instead
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 (the port Nginx listens on inside the container)
EXPOSE 80

# Command to run Nginx in the foreground when the container starts
CMD ["nginx", "-g", "daemon off;"]