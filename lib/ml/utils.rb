# frozen_string_literal: true

module ML
  module Utils
    class << self
      def make_blobs(n_samples: 100, n_features: 2, centers: 2, random_state: 0, reshape_y: true)
        x = Array.new(n_samples) { Array.new(n_features, 0) }
        y = Array.new(n_samples, 0)

        centers = Array.new(centers) { Array.new(n_features) { rand(-10..10) } }

        n_samples.times do |i|
          c = rand(0..centers.length - 1)
          n_features.times { |j| x[i][j] = rand(-10.0..10.0) + centers[c][j] }
          y[i] = c
        end

        y = reshape(y) if reshape_y

        [x, y]
      end

      def randn
        u1 = 2 * rand - 1
        u2 = 2 * rand - 1
        s  = u1**2 + u2**2

        return randn if s >= 1

        s = Math.sqrt(-2 * Math.log(s) / s)
        u1 * s
      end

      def reshape(y_data, slice = 1)
        y_data.flatten.each_slice(slice).to_a
      end

      def shapes(matrix)
        [matrix.length, matrix[0].is_a?(Array) ? matrix[0].length : nil]
      end

      def dot(x_data, w_data)
        x_data.map { |row| [row.map.with_index { |element, index| element * (w_data[index][0].is_a?(Array) ? w_data[index][0][0] : w_data[index][0]) }.reduce(&:+)] }
      end

      def accuracy_score(y_true, y_pred)
        total = y_true.length

        total.times.map { |i| y_true[i][0][0] == (y_pred[i][0] ? 1 : 0) ? 1 : 0 }.reduce(&:+) / total
      end

      def transposed_matrix(matrix)
        matrix[0]
          .length
          .times
          .map { |i| matrix.length.times.map { |j| matrix[j][i] } }
      end
    end
  end
end
