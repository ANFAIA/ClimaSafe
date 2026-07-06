# agents/workspace/

Aquí es donde los agentes escriben lo que generan — nunca en la raíz del
proyecto. Cada agente tiene su propia subcarpeta, creada automáticamente por
`SharedContext.agent_workspace("nombre_del_agente")` la primera vez que la
necesita:

```
agents/workspace/
├── notebook/     # manifests + imágenes PNG extraídas por NotebookAgent
├── installer/    # staging de repos clonados por InstallerAgent antes de moverlos a agents/external/
└── ...
```

Por defecto todo aquí se ignora en git (ver `.gitignore` de esta carpeta) —
es contenido derivado, regenerable en cualquier momento volviendo a
ejecutar el agente correspondiente.
