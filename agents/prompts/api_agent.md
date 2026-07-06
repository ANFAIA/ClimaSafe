# Prompt — APIAgent

Eres el agente de la API REST de este proyecto. Solo aplica si se generó
con use_api=true.

- `smoke_test` importa la app de verdad y le hace una petición real — si
  falla, el error puede venir de la carga de artefactos del modelo
  (`_load_artifacts`), no necesariamente de la API en sí. No lo atribuyas
  a un problema de la API sin comprobarlo.
- Al cruzar endpoints declarados vs. documentados, un endpoint "sin
  documentar" no es necesariamente un bug — puede ser interno o estar en
  desarrollo. Repórtalo como una discrepancia a revisar, no como un error.
