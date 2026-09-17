import hashlib
import json
import math
import numbers

NAMES = ('train', 'validation', 'test')


def scalar_key(value):
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bool):
        return json.dumps(['bool', value])
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            raise ValueError('Nonfinite class/group labels are unsupported')
        # Normalize 1 and 1.0 across nullable numeric batches without rounding large ints.
        return json.dumps(['number', str(int(value)) if value == int(value) else repr(float(value))])
    if isinstance(value, str):
        return json.dumps(['string', value], ensure_ascii=False)
    raise ValueError('Class/group labels must be scalar strings, booleans, or finite numbers')


def priority(seed, key):
    return hashlib.sha256(f'{seed}\0{key}'.encode()).hexdigest()


def allocate(n, ratios):
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if n < len(active):
        raise ValueError('Too few rows per split/class; reduce active splits or collect more examples')
    quotas = [n*r for r in ratios]
    counts = [int(q) for q in quotas]
    for i in sorted(active, key=lambda i: (-(quotas[i]-counts[i]), i))[:n-sum(counts)]:
        counts[i] += 1
    for i in active:
        if counts[i] == 0:
            donor = max(active, key=lambda j: counts[j]-quotas[j] if counts[j] > 1 else -math.inf)
            counts[donor] -= 1
            counts[i] += 1
    return counts


def assign(connection, config):
    total = connection.execute('SELECT count(*) FROM rows WHERE eligible=1').fetchone()[0]
    if not total:
        raise ValueError('No rows with observed targets')
    if config.strategy in {'random', 'stratified'}:
        groups = [(None, total)] if config.strategy == 'random' else connection.execute(
            'SELECT label,count(*) FROM rows WHERE eligible=1 GROUP BY label ORDER BY label').fetchall()
        for label, n in groups:
            counts = allocate(n, config.ratios)
            query = 'SELECT row_id FROM rows WHERE eligible=1'
            args = ()
            if label is not None:
                query += ' AND label=?'
                args = (label,)
            cursor = connection.execute(query+' ORDER BY priority,row_id', args)
            index, pending = 0, []
            for (row_id,) in cursor:
                split = NAMES[0] if index < counts[0] else NAMES[1] if index < sum(counts[:2]) else NAMES[2]
                pending.append((split, row_id))
                index += 1
                if len(pending) >= 10_000:
                    connection.executemany('UPDATE rows SET split=? WHERE row_id=?', pending)
                    pending.clear()
            connection.executemany('UPDATE rows SET split=? WHERE row_id=?', pending)
    elif config.strategy == 'group':
        cursor = connection.execute('SELECT DISTINCT group_key FROM rows WHERE eligible=1 ORDER BY group_key')
        for (key,) in cursor:
            fraction = int(priority(config.seed, key)[:16], 16)/2**64
            split = 'train' if fraction < config.train else 'validation' if fraction < config.train+config.validation else 'test'
            connection.execute('UPDATE rows SET split=? WHERE eligible=1 AND group_key=?', (split, key))
    else:
        seen = 0
        for timestamp, n in connection.execute('SELECT time_key,count(*) FROM rows WHERE eligible=1 GROUP BY time_key ORDER BY time_key'):
            # Keep identical timestamps together; boundary sizes can differ from requested ratios.
            fraction = seen/total
            split = 'train' if fraction < config.train else 'validation' if fraction < config.train+config.validation else 'test'
            connection.execute('UPDATE rows SET split=? WHERE eligible=1 AND time_key=?', (split, timestamp))
            seen += n
    connection.commit()
    counts = dict(connection.execute('SELECT split,count(*) FROM rows GROUP BY split'))
    for name, ratio in zip(NAMES, config.ratios):
        if ratio and not counts.get(name, 0):
            raise ValueError(f'{name} is empty; adjust fractions/seed or provide more groups/timestamps')
    return {name: counts.get(name, 0) for name in (*NAMES, 'excluded')}
