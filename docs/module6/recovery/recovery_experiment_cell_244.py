from pathlib import Path
import ast
import hashlib
import inspect
import subprocess
ROOT = Path('/content/adversarially-hardened-dynamic-sensing')
STATE = AHDS_M6_HELDOUT_STATE
CHECKPOINT = 'a9258ba4a307d6a6ad4ff23788a392c16260ed48'
assert STATE['task_passed']
assert len(STATE['protected_files']) == 135
assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == CHECKPOINT
print('=' * 70)
print('AHDS — M6-DEV-01D-01')
print('Integrated Perception Interface Preflight')
print('=' * 70)
paths = ['src/ahds/integration/closed_loop_controller.py', 'src/ahds/integration/joint_closed_loop_controller.py', 'src/ahds/integration/roi_closed_loop.py', 'src/ahds/module1_environment/optical_environment.py']
for relative_path in paths:
    path = ROOT / relative_path
    assert path.is_file(), relative_path
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if relative_path in STATE['protected_files']:
        assert digest == STATE['protected_files'][relative_path]
        verification = 'Protected inventory'
    else:
        committed_source = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{CHECKPOINT}:{relative_path}'])
        assert source == committed_source, relative_path
        verification = 'Committed Git checkpoint'
    print('\n' + '=' * 70)
    print(path.name)
    print('Source verification:', verification, '— PASS')
    print('=' * 70)
    tree = ast.parse(source.decode('utf-8'))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            print(f'\nclass {node.name}:')
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    print(f'  {item.name}({ast.unparse(item.args)})')
        elif isinstance(node, ast.FunctionDef):
            print(f'\ndef {node.name}({ast.unparse(node.args)})')
print('\n[FROZEN PERCEPTION MODEL]')
print('Model:', type(FROZEN_MODEL).__name__)
print('Training mode:', FROZEN_MODEL.training)
print('Forward:', inspect.signature(FROZEN_MODEL.forward))
assert not FROZEN_MODEL.training
assert all((not p.requires_grad for p in FROZEN_MODEL.parameters()))
model_path = ROOT / 'results/module6/model_training/best_model.pt'
assert hashlib.sha256(model_path.read_bytes()).hexdigest() == '5a4d3736aef265bce70dc540cad07a83e4c4555244c2185806478507cdce257f'
print('Frozen checkpoint identity: PASS')
print('\nProtected inventory:', len(STATE['protected_files']))
print('Source modifications: NONE')
print('Model modifications: NONE')
print('M6-DEV-01D-01: INTERFACE PREFLIGHT PASSED')