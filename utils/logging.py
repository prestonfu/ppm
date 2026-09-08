def prefix_metrics(metrics, prefix):
    return {prefix + k: v for k, v in sorted(metrics.items())}


def wandb_table_from_rows(rows):
    import wandb

    columns = sorted(dict.fromkeys(key for row in rows for key in row))
    return wandb.Table(columns=columns, data=[[row.get(column, '') for column in columns] for row in rows])


def crossed_multiple(previous, current, every):
    return every > 0 and previous // every < current // every
