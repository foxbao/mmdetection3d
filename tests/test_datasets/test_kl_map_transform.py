from pathlib import Path

import numpy as np

from projects.KL8.map_utils import load_kl_base_map, rasterize_local_map
from projects.KL8.transforms import GenerateKLMapMask


MINI_MAP = """road {
  id {
    id: "road_1"
  }
  section {
    id {
      id: "section_1"
    }
    boundary {
      outer_polygon {
        edge {
          curve {
            segment {
              line_segment {
                point {
                  x: -4
                  y: 1
                }
                point {
                  x: 4
                  y: 1
                }
              }
            }
          }
        }
        edge {
          curve {
            segment {
              line_segment {
                point {
                  x: -4
                  y: -1
                }
                point {
                  x: 4
                  y: -1
                }
              }
            }
          }
        }
      }
    }
  }
}
junction {
  id {
    id: "junction_1"
  }
  polygon {
    point {
      x: 4
      y: -1
    }
    point {
      x: 6
      y: -1
    }
    point {
      x: 6
      y: 1
    }
    point {
      x: 4
      y: 1
    }
    point {
      x: 4
      y: -1
    }
  }
}
lane {
  id {
    id: "lane_1"
  }
  central_curve {
    segment {
      line_segment {
        point {
          x: -4
          y: 0
        }
        point {
          x: 4
          y: 0
        }
      }
    }
  }
  left_boundary {
    curve {
      segment {
        line_segment {
          point {
            x: -4
            y: 0.5
          }
          point {
            x: 4
            y: 0.5
          }
        }
      }
    }
  }
  right_boundary {
    curve {
      segment {
        line_segment {
          point {
            x: -4
            y: -0.5
          }
          point {
            x: 4
            y: -0.5
          }
        }
      }
    }
  }
}
"""


def test_load_and_rasterize_kl_map(tmp_path: Path):
    map_file = tmp_path / 'base_map.txt'
    map_file.write_text(MINI_MAP, encoding='utf-8')

    map_data = load_kl_base_map(map_file)
    assert len(map_data['roads']) == 1
    assert len(map_data['junctions']) == 1
    assert len(map_data['lanes']) == 1
    assert map_data['lanes'][0]['polygon'].shape[1] == 2

    local_map = dict(
        roads=map_data['roads'],
        junctions=map_data['junctions'],
        lanes=map_data['lanes'])
    masks = rasterize_local_map(
        local_map=local_map,
        point_cloud_range=[-8, -4, -1, 8, 4, 1],
        mask_shape=(32, 64))
    assert masks['road'].sum() > 0
    assert masks['junction'].sum() > 0
    assert masks['lane'].sum() > 0
    assert masks['drivable'].sum() >= masks['road'].sum()


def test_generate_kl_map_mask_transform(tmp_path: Path):
    map_file = tmp_path / 'base_map.txt'
    map_file.write_text(MINI_MAP, encoding='utf-8')

    transform = GenerateKLMapMask(
        map_file=str(map_file),
        point_cloud_range=[-8, -4, -1, 8, 4, 1],
        mask_shape=(32, 64),
        target='drivable')

    results = transform(dict(ego2global=np.eye(4, dtype=np.float32)))
    assert 'gt_seg_map' in results
    assert results['gt_seg_map'].shape == (32, 64)
    assert results['gt_seg_map'].dtype == np.uint8
    assert results['gt_seg_map'].sum() > 0
