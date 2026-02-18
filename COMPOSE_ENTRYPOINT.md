# Optional: entrypoint to fix OpenCV libs without rebuilding

If the container fails with `libgthread-2.0.so.0` (or similar) and you don't want to rebuild the image, use the entrypoint script to install the missing libs at startup.

1. Ensure `entrypoint.sh` is at the repo root. After `git pull` it is at `./entrypoint.sh` (stack root = repo root).

2. Make it executable on the server:
   ```bash
   chmod +x /mnt/user/appdata/dockge/stacks/frigate-buffer/entrypoint.sh
   ```

3. In your `docker-compose` file, add the volume and override entrypoint/command:

   ```yaml
   services:
     frigate-buffer:
       image: frigate-buffer:latest
       # ... keep your existing build, volumes, environment, etc. ...
       volumes:
         - /mnt/user/appdata/frigate_buffer:/app/storage
         - /mnt/user/appdata/frigate_buffer/config.yaml:/app/config.yaml:ro
         - /etc/localtime:/etc/localtime:ro
         - ./entrypoint.sh:/entrypoint.sh:ro   # add this
       entrypoint: ["/entrypoint.sh"]               # add this
       command: ["python", "-m", "frigate_buffer.main"]   # add this
   ```

4. Start the stack: `docker compose up -d --force-recreate` (or via Dockge).

The first start will run `apt-get install` (requires network); later starts are quick. Once you rebuild the image with the full OpenCV deps in the Dockerfile, you can remove the entrypoint volume and the `entrypoint`/`command` overrides.
